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
 * APPLY_ALL is the important v2 change. It uses:
 *   entry.readImageData()
 *   ImageDisplay.create(imageData)
 *   entry.saveImageData(imageData)
 *
 * It intentionally does not use qupath.openImageEntry(), viewer.setImageData(),
 * or getCurrentViewer() inside the project loop.
 */

// ===================== USER SETTINGS =====================

def MODE = "CAPTURE"          // "CAPTURE", "APPLY", or "APPLY_ALL"
def PRESET_NAME = "IY-test"   // saved_display_presets/${PRESET_NAME}.json

// EXACT assumes matching full display channel names between source and target.
// MARKER keeps the old marker-extraction behavior as an escape hatch.
def MATCH_MODE = "EXACT"      // "EXACT" or "MARKER"

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

def rgbToHex = { Integer packed ->
    if (packed == null)
        return null
    int r = ColorTools.red(packed)
    int g = ColorTools.green(packed)
    int b = ColorTools.blue(packed)
    return String.format("#%02X%02X%02X", r, g, b)
}

def hexToPackedRGB = { String hex ->
    if (hex == null || hex.trim().isEmpty())
        return null
    hex = hex.trim().replace("#", "")
    if (hex.length() != 6)
        throw new IllegalArgumentException("Expected #RRGGBB color, got '${hex}'")
    int r = Integer.parseInt(hex.substring(0, 2), 16)
    int g = Integer.parseInt(hex.substring(2, 4), 16)
    int b = Integer.parseInt(hex.substring(4, 6), 16)
    return ColorTools.packRGB(r, g, b)
}

def stripDisplaySuffix = { String name ->
    return (name ?: "").replaceFirst(/\s*\(C\d+\)$/, "")
}

def extractMarker = { String channelName ->
    def base = stripDisplaySuffix(channelName)
    def match = (base =~ /NK_\w+_\w+_C\d+R\d+_(.+)$/)
    if (match.matches())
        return match.group(1).toUpperCase()
    return base.toUpperCase()
}

def settingKeyForChannel = { ch ->
    def name = ch.getName()
    if (MATCH_MODE.toUpperCase() == "MARKER")
        return extractMarker(name)
    return name
}

def getSettingForChannel = { ch, Map channels ->
    def mode = MATCH_MODE.toUpperCase()
    def name = ch.getName()

    if (mode == "MARKER")
        return channels[extractMarker(name)]

    // Exact display-name match first. The fallback handles metadata names that
    // omit QuPath's display suffix, e.g. "Marker" vs "Marker (C1)".
    def s = channels[name]
    if (s == null)
        s = channels[stripDisplaySuffix(name)]
    return s
}

def getChannelIndex = { ch, int fallbackIndex ->
    try {
        return ch.getChannel() as int
    } catch (Throwable ignored) {
        return fallbackIndex
    }
}

def captureFromDisplay = { display, imageName ->
    def selectedNames = display.selectedChannels().collect { it.getName() } as Set
    def markerCounts = [:]

    if (MATCH_MODE.toUpperCase() == "MARKER") {
        display.availableChannels().each { ch ->
            def m = extractMarker(ch.getName())
            markerCounts[m] = (markerCounts[m] ?: 0) + 1
        }
    }

    def channelSettings = [:]
    display.availableChannels().each { ch ->
        def key = settingKeyForChannel(ch)
        if (MATCH_MODE.toUpperCase() == "MARKER" && markerCounts[key] > 1)
            key = ch.getName().toUpperCase()

        channelSettings[key] = [
            name       : ch.getName(),
            serverName : stripDisplaySuffix(ch.getName()),
            min        : ch.getMinDisplay(),
            max        : ch.getMaxDisplay(),
            show       : selectedNames.contains(ch.getName()),
            color      : rgbToHex(ch.getColor())
        ]
    }

    return [
        version  : 2,
        source   : imageName,
        matchMode: MATCH_MODE.toUpperCase(),
        channels : channelSettings
    ]
}

def applyToDisplay = { display, Map preset ->
    int applied = 0
    int skipped = 0
    def matched = []

    display.availableChannels().eachWithIndex { ch, idx ->
        def s = getSettingForChannel(ch, preset.channels as Map)
        if (s == null) {
            skipped++
        } else {
            display.setMinMaxDisplay(ch, (s.min as Number).floatValue(), (s.max as Number).floatValue())
            matched << [channel: ch, setting: s, index: getChannelIndex(ch, idx)]
            applied++
        }
    }

    // Apply visibility after min/max so additive fluorescence channels end up
    // with the requested final selected-channel set.
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

def applyColorsToImageData = { imageData, List matched ->
    def metadata = imageData.getServerMetadata()
    def oldChannels = metadata.getChannels()
    def newChannels = new ArrayList(oldChannels)
    boolean changed = false

    matched.each { item ->
        def s = item.setting
        if (s.color != null) {
            int idx = item.index as int
            if (idx >= 0 && idx < oldChannels.size()) {
                Integer newColor = hexToPackedRGB(s.color as String)
                if (newColor != null) {
                    def oldCh = oldChannels[idx]
                    if (oldCh.getColor() != newColor) {
                        newChannels[idx] = ImageChannel.getInstance(oldCh.getName(), newColor)
                        changed = true
                    }
                }
            }
        }
    }

    if (changed) {
        def newMetadata = new ImageServerMetadata.Builder(metadata)
            .channels(newChannels)
            .build()
        imageData.updateServerMetadata(newMetadata)
    }
    return changed
}

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
    def backupDir = new File(srcProjectDir, "backups/pre_transfer_display_v2_${timestamp}")
    backupDir.mkdirs()

    srcProjectDir.listFiles()?.each { child ->
        if (child.getName() != "backups")
            copyRecursively(child, new File(backupDir, child.getName()))
    }
    return backupDir
}

def loadPreset = {
    if (!presetFile.exists()) {
        println "ERROR: No preset named '${PRESET_NAME}' found."
        println "       Expected file: ${presetFile.getAbsolutePath()}"
        println "       Run CAPTURE first."
        return null
    }
    def preset = gson.fromJson(presetFile.getText("UTF-8"), mapType)
    if (!(preset?.channels instanceof Map)) {
        println "ERROR: Preset file does not contain a channels map: ${presetFile.getAbsolutePath()}"
        return null
    }
    return preset
}

def mode = MODE.toUpperCase()
def matchMode = MATCH_MODE.toUpperCase()
if (!(matchMode in ["EXACT", "MARKER"])) {
    println "ERROR: MATCH_MODE must be EXACT or MARKER; currently '${MATCH_MODE}'."
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
        println "  [${flag}] ${key}: ${val.min} - ${val.max} ${val.color}"
    }

} else if (mode == "APPLY") {
    def viewer = getCurrentViewer()
    if (viewer == null) {
        println "ERROR: No image is open. Open the target slide first, then run APPLY."
        return
    }

    def preset = loadPreset()
    if (preset == null)
        return

    def imageData = getCurrentImageData()
    def display = viewer.getImageDisplay()
    def result = applyToDisplay(display, preset)
    def colorsChanged = applyColorsToImageData(imageData, result.matched as List)
    if (colorsChanged) {
        display.refreshChannelOptions()
        display.saveChannelColorProperties()
    }
    viewer.repaint()

    println "=== APPLY COMPLETE ==="
    println "Preset       : ${PRESET_NAME}"
    println "Target       : ${imageData.getServerMetadata().getName()}"
    println "Updated      : ${result.applied}"
    println "Skipped      : ${result.skipped}"
    println "Colors       : ${colorsChanged ? 'updated' : 'unchanged'}"
    println "NOTE: Save the current image/project in QuPath if you want this single-image APPLY persisted."

} else if (mode == "APPLY_ALL") {
    def preset = loadPreset()
    if (preset == null)
        return

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

    project.getImageList().eachWithIndex { entry, entryIndex ->
        def label = "${entryIndex + 1}/${project.getImageList().size()} ${entry.getImageName()}"
        try {
            def imageData = entry.readImageData()
            def display = ImageDisplay.create(imageData)

            def result = applyToDisplay(display, preset)
            def colorsChanged = applyColorsToImageData(imageData, result.matched as List)
            if (colorsChanged) {
                display.refreshChannelOptions()
                display.saveChannelColorProperties()
                totalColorChanged++
            }

            if (!DRY_RUN) {
                entry.saveImageData(imageData)
                totalSaved++
            }

            totalApplied += result.applied as int
            totalSkipped += result.skipped as int
            println "${label}"
            println "  updated ${result.applied}, skipped ${result.skipped}, colors ${colorsChanged ? 'updated' : 'unchanged'}"

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
