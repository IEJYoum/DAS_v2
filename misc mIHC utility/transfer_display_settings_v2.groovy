/*
 * transfer_display_settings_v2.groovy
 *
 * QuPath 0.7.0 display settings transfer for multiplex fluorescence projects.
 *
 * Modes:
 *   CAPTURE   - Capture the currently open viewer's display settings to JSON.
 *   APPLY     - Apply a captured preset to the currently open viewer/image.
 *   APPLY_ALL - Apply a captured preset to every project entry without opening
 *               images in the viewer.
 *
 * APPLY_ALL uses headless ImageDisplay (no viewer loop):
 *   entry.readImageData()
 *   ImageDisplay.create(imageData)
 *   entry.saveImageData(imageData)
 *
 * It intentionally does NOT use qupath.openImageEntry(), viewer.setImageData(),
 * or getCurrentViewer() inside the project loop.
 *
 * Match modes:
 *   MARKER    - (default) Strips everything up to the _C{cycle}R{round}_
 *               delimiter and matches on the remaining suffix.
 *               Works across any prefix convention (NK_, KB_UCSD_, etc.).
 *   FULL_NAME - Matches on the full QuPath channel name. Only useful when
 *               source and target have identical channel names.
 */

// ===================== USER SETTINGS =====================

def MODE = "APPLY_ALL"          // "CAPTURE", "APPLY", or "APPLY_ALL"
def PRESET_NAME = "IY-test1"   // saved_display_presets/${PRESET_NAME}.json

// MARKER (default): strip slide-specific prefix, match on marker suffix.
//   NK_3xTLS_403932_C02R2_CD3_fixed  ->  CD3_FIXED
//   NK_SCREEN_B171_C02R2_CD3_fixed   ->  CD3_FIXED   (matches)
// FULL_NAME: match on full channel name (same-slide or identical naming only).
def MATCH_MODE = "MARKER"     // "MARKER" or "FULL_NAME"

def BACKUP_BEFORE_APPLY_ALL = true
def DRY_RUN = false           // If true, APPLY_ALL reports what it would do but saves nothing.

// =================== END USER SETTINGS ===================


import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.reflect.TypeToken
import qupath.lib.common.ColorTools
import qupath.lib.display.ImageDisplay
import qupath.lib.images.servers.ImageChannel
import qupath.lib.images.servers.ImageServerMetadata

import java.nio.file.Files
import java.nio.file.StandardCopyOption

def gson = new GsonBuilder().setPrettyPrinting().create()
def mapType = new TypeToken<Map<String, Object>>(){}.getType()

def project = getProject()
if (project == null) {
    println "ERROR: No project is open."
    return
}

def projectFile = project.getPath().toFile()
def projectDir = projectFile.getParentFile()
def presetsDir = new File(projectDir, "saved_display_presets")
presetsDir.mkdirs()
def presetFile = new File(presetsDir, "${PRESET_NAME}.json")


// =================== HELPER CLOSURES ===================

def rgbToHex = { Integer packed ->
    if (packed == null) return null
    int r = ColorTools.red(packed)
    int g = ColorTools.green(packed)
    int b = ColorTools.blue(packed)
    return String.format("#%02X%02X%02X", r, g, b)
}

def hexToPackedRGB = { String hex ->
    if (hex == null || hex.trim().isEmpty()) return null
    hex = hex.trim().replace("#", "")
    if (hex.length() != 6)
        throw new IllegalArgumentException("Expected #RRGGBB color, got '${hex}'")
    int r = Integer.parseInt(hex.substring(0, 2), 16)
    int g = Integer.parseInt(hex.substring(2, 4), 16)
    int b = Integer.parseInt(hex.substring(4, 6), 16)
    return ColorTools.packRGB(r, g, b)
}

/** Remove QuPath's display suffix like " (C1)" from a channel name */
def stripDisplaySuffix = { String name ->
    return (name ?: "").replaceFirst(/\s*\(C\d+\)$/, "")
}

/**
 * Extract marker suffix from a full channel name.
 * Strips everything up to and including the _C{cycle}R{round}_ delimiter,
 * regardless of the project-specific prefix.
 *
 *   "NK_3xTLS_403932_C02R2_CD3_fixed (C1)"       -> "CD3_FIXED"
 *   "NK_SCREEN_B171_C02R2_CD3_fixed"              -> "CD3_FIXED"
 *   "KB_UCSD_FCPDAC005_D30_C03R1_CD3_fixed"       -> "CD3_FIXED"
 *   "NK_3xTLS_403932_C01R1_B220"                  -> "B220"
 *   "SomeOtherFormat"                             -> "SOMEOTHERFORMAT" (fallback)
 */
def extractMarker = { String channelName ->
    def base = stripDisplaySuffix(channelName)
    def match = (base =~ /.*_C\d+R\d+_(.+)$/)
    if (match.matches())
        return match.group(1).toUpperCase()
    return base.toUpperCase()
}

/**
 * Build a marker key for a channel, handling duplicates with numbered suffixes.
 * markerCounts: map of marker -> count (pre-computed)
 * markerSeen:   map of marker -> times seen so far (mutated)
 * Returns e.g. "CD3_FIXED", or "HEM" / "HEM__2" for duplicates.
 */
def markerKey = { String channelName, Map markerCounts, Map markerSeen ->
    def marker = extractMarker(channelName)
    if (markerCounts[marker] > 1) {
        markerSeen[marker] = (markerSeen[marker] ?: 0) + 1
        int n = markerSeen[marker]
        return n == 1 ? marker : "${marker}__${n}"
    }
    return marker
}

/**
 * Count how many times each marker appears in a list of channel names.
 */
def countMarkers = { List<String> names ->
    def counts = [:]
    names.each { name ->
        def m = extractMarker(name)
        counts[m] = (counts[m] ?: 0) + 1
    }
    return counts
}

/**
 * Look up preset settings for a channel.
 * In MARKER mode: extract marker, compute numbered key, look up.
 * In FULL_NAME mode: try full name, then stripped name.
 */
def getSettingForChannel = { String channelName, Map channels, Map markerCounts, Map markerSeen ->
    if (MATCH_MODE.toUpperCase() == "MARKER") {
        def key = markerKey(channelName, markerCounts, markerSeen)
        return [setting: channels[key], key: key]
    }
    // FULL_NAME mode
    def s = channels[channelName]
    if (s != null) return [setting: s, key: channelName]
    def stripped = stripDisplaySuffix(channelName)
    s = channels[stripped]
    return [setting: s, key: stripped]
}


// =================== CAPTURE ===================

def captureFromDisplay = { display, String imageName ->
    def available = display.availableChannels()
    def selectedNames = display.selectedChannels().collect { it.getName() } as Set

    def names = available.collect { it.getName() }
    def markerCounts = countMarkers(names)
    def markerSeen = [:]

    def channelSettings = new LinkedHashMap()
    available.each { ch ->
        def chName = ch.getName()
        String key
        if (MATCH_MODE.toUpperCase() == "MARKER") {
            key = markerKey(chName, markerCounts, markerSeen)
        } else {
            key = chName
        }

        channelSettings[key] = [
            name      : chName,
            min       : Math.round(ch.getMinDisplay() * 100.0) / 100.0,
            max       : Math.round(ch.getMaxDisplay() * 100.0) / 100.0,
            show      : selectedNames.contains(chName),
            color     : rgbToHex(ch.getColor())
        ]
    }

    return [
        version  : 2,
        source   : imageName,
        matchMode: MATCH_MODE.toUpperCase(),
        channels : channelSettings
    ]
}


// =================== APPLY DISPLAY SETTINGS ===================

def applyToDisplay = { display, Map preset ->
    def available = display.availableChannels()
    int applied = 0
    int skipped = 0

    // Pre-compute marker counts for this image's channels
    def names = available.collect { it.getName() }
    def markerCounts = countMarkers(names)
    def markerSeen = [:]

    // First pass: apply min/max and collect matches
    def matched = []
    def unmatched = []
    available.each { ch ->
        def result = getSettingForChannel(ch.getName(), preset.channels as Map, markerCounts, markerSeen)
        def s = result.setting
        if (s == null) {
            println "    [off]  ${ch.getName()}  (${result.key})  -- no match, hidden"
            unmatched << ch
            skipped++
        } else {
            display.setMinMaxDisplay(ch, (s.min as Number).floatValue(), (s.max as Number).floatValue())
            matched << [channel: ch, setting: s, key: result.key]
            applied++
        }
    }

    // Second pass: apply visibility (deselect first, then select, to avoid
    // QuPath's auto-deselection behavior in non-additive mode)
    unmatched.each { ch -> display.setChannelSelected(ch, false) }
    matched.each { item ->
        if (!(item.setting.show as boolean))
            display.setChannelSelected(item.channel, false)
    }
    matched.each { item ->
        if (item.setting.show as boolean)
            display.setChannelSelected(item.channel, true)
    }

    return [applied: applied, skipped: skipped, matched: matched]
}


// =================== APPLY COLORS VIA METADATA ===================

/**
 * Apply colors by iterating metadata channels directly and matching by
 * marker key (name-based, not index-based). This is safe even if
 * ImageDisplay reorders channels.
 */
def applyColorsToImageData = { imageData, Map preset ->
    def metadata = imageData.getServerMetadata()
    def oldChannels = metadata.getChannels()
    boolean changed = false

    // Count markers in metadata channel list
    def metaNames = oldChannels.collect { it.getName() }
    def markerCounts = countMarkers(metaNames)
    def markerSeen = [:]

    def newChannels = oldChannels.collect { ch ->
        def result = getSettingForChannel(ch.getName(), preset.channels as Map, markerCounts, markerSeen)
        def s = result.setting
        if (s != null && s.color != null) {
            Integer newColor = hexToPackedRGB(s.color as String)
            if (newColor != null && ch.getColor() != newColor) {
                changed = true
                return ImageChannel.getInstance(ch.getName(), newColor)
            }
        }
        return ch
    }

    if (changed) {
        def newMetadata = new ImageServerMetadata.Builder(metadata)
            .channels(newChannels)
            .build()
        imageData.updateServerMetadata(newMetadata)
    }
    return changed
}


// =================== BACKUP ===================

def copyRecursively
copyRecursively = { File src, File dst ->
    if (src.isDirectory()) {
        dst.mkdirs()
        src.listFiles()?.each { child ->
            copyRecursively(child, new File(dst, child.getName()))
        }
    } else {
        dst.getParentFile().mkdirs()
        Files.copy(src.toPath(), dst.toPath(), StandardCopyOption.REPLACE_EXISTING)
    }
}

def backupProject = { File srcProjectDir ->
    def timestamp = new java.text.SimpleDateFormat("yyyy-MM-dd_HHmmss").format(new Date())
    def backupDir = new File(srcProjectDir, "backups/pre_v2_${timestamp}")
    backupDir.mkdirs()

    // Back up project file and data/ only (not resources/, classifiers/, etc.)
    def projectF = new File(srcProjectDir, "project.qpproj")
    if (projectF.exists())
        Files.copy(projectF.toPath(), new File(backupDir, "project.qpproj").toPath())
    def projectBackup = new File(srcProjectDir, "project.qpproj.backup")
    if (projectBackup.exists())
        Files.copy(projectBackup.toPath(), new File(backupDir, "project.qpproj.backup").toPath())
    def dataDir = new File(srcProjectDir, "data")
    if (dataDir.isDirectory())
        copyRecursively(dataDir, new File(backupDir, "data"))

    return backupDir
}


// =================== PRESET LOADING ===================

def loadPreset = {
    if (!presetFile.exists()) {
        println "ERROR: No preset named '${PRESET_NAME}' found."
        println "       Expected file: ${presetFile.getAbsolutePath()}"
        println "       Run CAPTURE first."
        return null
    }
    def preset = gson.fromJson(presetFile.getText("UTF-8"), mapType)
    if (!(preset?.channels instanceof Map)) {
        println "ERROR: Preset file does not contain a channels map."
        return null
    }
    return preset
}


// =================== MAIN ===================

def mode = MODE.toUpperCase()
def matchMode = MATCH_MODE.toUpperCase()
if (!(matchMode in ["MARKER", "FULL_NAME"])) {
    println "ERROR: MATCH_MODE must be MARKER or FULL_NAME; currently '${MATCH_MODE}'."
    return
}


if (mode == "CAPTURE") {

    def viewer = getCurrentViewer()
    if (viewer == null) {
        println "ERROR: No image is open. Open the source slide first, then run CAPTURE."
        return
    }

    def imageData = getCurrentImageData()
    def imageName = imageData.getServerMetadata().getName()
    def display = viewer.getImageDisplay()
    def preset = captureFromDisplay(display, imageName)
    presetFile.setText(gson.toJson(preset), "UTF-8")

    println "=== CAPTURE COMPLETE ==="
    println "Source slide : ${imageName}"
    println "Preset name  : ${PRESET_NAME}"
    println "Match mode   : ${matchMode}"
    println "Saved to     : ${presetFile.getAbsolutePath()}"
    println "Channels     : ${preset.channels.size()}"
    println ""
    preset.channels.each { key, val ->
        def flag = (val.show as boolean) ? "ON " : "off"
        println "  [${flag}]  ${key}:  ${val.min} - ${val.max}  ${val.color}"
    }
    println ""
    println "Next steps:"
    println "  1. Change MODE to \"APPLY\" or \"APPLY_ALL\""
    println "  2. Run the script again"


} else if (mode == "APPLY") {

    def viewer = getCurrentViewer()
    if (viewer == null) {
        println "ERROR: No image is open. Open the target slide first, then run APPLY."
        return
    }

    def preset = loadPreset()
    if (preset == null) return

    def imageData = getCurrentImageData()
    def display = viewer.getImageDisplay()

    println "=== APPLYING PRESET '${PRESET_NAME}' ==="
    println "Settings from : ${preset.source}"
    println "Applying to   : ${imageData.getServerMetadata().getName()}"
    println ""

    // Apply colors first, then refresh, then display settings (min/max, visibility).
    // refreshChannelOptions() rebuilds channel objects, so display settings must
    // be applied AFTER it runs or they get wiped.
    def colorsChanged = applyColorsToImageData(imageData, preset)
    if (colorsChanged) {
        display.refreshChannelOptions()
    }
    def result = applyToDisplay(display, preset)

    // Persist display state (min/max, visibility, colors) into imageData properties
    display.saveChannelColorProperties()
    viewer.repaint()

    println ""
    println "=== APPLY COMPLETE ==="
    println "Updated      : ${result.applied}"
    println "Skipped      : ${result.skipped}"
    println "Colors       : ${colorsChanged ? 'updated' : 'unchanged'}"


} else if (mode == "APPLY_ALL") {

    def preset = loadPreset()
    if (preset == null) return

    println "=== APPLY_ALL START ==="
    println "Project      : ${projectFile.getAbsolutePath()}"
    println "Preset       : ${presetFile.getAbsolutePath()}"
    println "Match mode   : ${matchMode}"
    println "Dry run      : ${DRY_RUN}"
    println "Entries      : ${project.getImageList().size()}"
    println ""

    if (BACKUP_BEFORE_APPLY_ALL && !DRY_RUN) {
        def backupDir = backupProject(projectDir)
        println "Backup saved : ${backupDir.getAbsolutePath()}"
        println ""
    }

    int totalApplied = 0
    int totalSkipped = 0
    int totalColorChanged = 0
    int totalSaved = 0
    def failures = []

    def entries = project.getImageList()

    entries.eachWithIndex { entry, entryIndex ->
        if (Thread.currentThread().isInterrupted()) {
            println "\nScript stopped by user at image ${entryIndex}/${entries.size()}."
            return  // breaks out of closure
        }

        def label = "[${entryIndex + 1}/${entries.size()}] ${entry.getImageName()}"
        try {
            def imageData = entry.readImageData()
            def display = ImageDisplay.create(imageData)

            // Apply colors first, then refresh, then display settings.
            // refreshChannelOptions() rebuilds channel objects, so display
            // settings must be applied AFTER it runs or they get wiped.
            def colorsChanged = applyColorsToImageData(imageData, preset)
            if (colorsChanged) {
                display.refreshChannelOptions()
                totalColorChanged++
            }
            def result = applyToDisplay(display, preset)

            // Persist display state into imageData properties
            display.saveChannelColorProperties()

            if (!DRY_RUN) {
                entry.saveImageData(imageData)
                totalSaved++
            }

            totalApplied += result.applied as int
            totalSkipped += result.skipped as int
            println "${label}"
            println "  updated ${result.applied}, skipped ${result.skipped}, colors ${colorsChanged ? 'set' : 'unchanged'}"

        } catch (Throwable t) {
            failures << [entry: entry.getImageName(), error: t.toString()]
            println "${label}"
            println "  ERROR: ${t}"
        }
    }

    if (!DRY_RUN) {
        try {
            project.syncChanges()
        } catch (Throwable t) {
            println "WARNING: project.syncChanges() failed: ${t}"
        }
    }

    println ""
    println "=== APPLY_ALL COMPLETE ==="
    println "Images saved       : ${totalSaved}"
    println "Channel updates    : ${totalApplied}"
    println "Channel skips      : ${totalSkipped}"
    println "Color image count  : ${totalColorChanged}"
    println "Failures           : ${failures.size()}"
    if (!failures.isEmpty()) {
        failures.each { f ->
            println "  ${f.entry}: ${f.error}"
        }
    }

} else {
    println "ERROR: MODE must be CAPTURE, APPLY, or APPLY_ALL; currently '${MODE}'."
}
