/*
 * ============================================================
 *  TRANSFER CHANNEL DISPLAY SETTINGS BETWEEN SLIDES
 *  For: 3x Tumor Screen QuPath project
 * ============================================================
 *
 *  This script saves and applies display settings (brightness,
 *  contrast, channel visibility) across slides. It matches
 *  channels by marker name (CD3, B220, PANCK, etc.) rather
 *  than the full channel name, so settings transfer correctly
 *  even though each slide has a different naming prefix.
 *
 *  See "applying_channel_settings_instructions.txt" in this
 *  folder for step-by-step usage instructions.
 *
 * ============================================================
 */

// ===================== USER SETTINGS ========================
// Change MODE to "CAPTURE", "APPLY", or "APPLY_ALL"
//   CAPTURE   = save the current slide's display settings
//   APPLY     = load saved settings onto the current slide
//   APPLY_ALL = generate a QuPath display preset that works
//               on every slide in the project (see instructions)

def MODE = "APPLY"

// Name your preset so you can have multiple saved configurations.
// Examples: "default", "TLS_view", "immune_panel", "BCL6_focus"

def PRESET_NAME = "TLS_SS2"

// ===================== END USER SETTINGS ====================


import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.reflect.TypeToken
import qupath.lib.common.ColorTools
import qupath.lib.images.servers.ImageChannel
import qupath.lib.images.servers.ImageServerMetadata

def gson = new GsonBuilder().setPrettyPrinting().create()

/** Pack RGB components into a hex string like "#00FFFF" */
def rgbToHex(int packed) {
    def r = (packed >> 16) & 0xFF
    def g = (packed >> 8) & 0xFF
    def b = packed & 0xFF
    return String.format("#%02X%02X%02X", r, g, b)
}

/** Parse a hex string like "#00FFFF" into a packed RGB int */
def hexToPackedRGB(String hex) {
    hex = hex.replace("#", "")
    def r = Integer.parseInt(hex.substring(0, 2), 16)
    def g = Integer.parseInt(hex.substring(2, 4), 16)
    def b = Integer.parseInt(hex.substring(4, 6), 16)
    return ColorTools.packRGB(r, g, b)
}

/** Convert hex "#RRGGBB" to the {red, green, blue} map QuPath presets use */
def hexToRgbMap(String hex) {
    hex = hex.replace("#", "")
    return [
        red:   Integer.parseInt(hex.substring(0, 2), 16),
        green: Integer.parseInt(hex.substring(2, 4), 16),
        blue:  Integer.parseInt(hex.substring(4, 6), 16)
    ]
}

/**
 * Extracts the marker name from a full channel name.
 *   "NK_SCREEN_B171_C02R2_CD3_fixed (C1)" -> "CD3_FIXED"
 *   "NK_3xTLS_403932_C01R1_B220 (C2)"     -> "B220"
 *   "NK_SCREEN_B22_C06R2_BCL6"            -> "BCL6"
 */
def extractMarker(String channelName) {
    def match = (channelName =~ /NK_\w+_\w+_C\d+R\d+_(.+?)(?:\s*\(C\d+\))?$/)
    if (match.matches()) {
        return match.group(1).toUpperCase()
    }
    // Fallback: use the full channel name
    return channelName.toUpperCase()
}


// ============ MAIN SCRIPT LOGIC ============

def project = getProject()
if (project == null) {
    println "ERROR: No project is open."
    return
}

// APPLY_ALL only needs the project, not the viewer
def viewer = getCurrentViewer()
def imageDisplay = null
def availableChannels = null
def selectedChannels  = null
def imageName = null

if (MODE.toUpperCase() != "APPLY_ALL") {
    if (viewer == null) {
        println "ERROR: No image is open. Open a slide first, then re-run."
        return
    }
    imageDisplay = viewer.getImageDisplay()
    availableChannels = imageDisplay.availableChannels()
    selectedChannels  = imageDisplay.selectedChannels()
    imageName = getCurrentImageData().getServer().getMetadata().getName()
}

// Presets are saved inside the project folder
def projectDir = project.getPath().getParent().toFile()
def presetsDir = new File(projectDir, "saved_display_presets")
presetsDir.mkdirs()
def presetFile = new File(presetsDir, "${PRESET_NAME}.json")


if (MODE.toUpperCase() == "CAPTURE") {

    // --- Count duplicate marker names (e.g. multiple HEM channels) ---
    def markerCounts = [:]
    availableChannels.each { ch ->
        def m = extractMarker(ch.getName())
        markerCounts[m] = (markerCounts[m] ?: 0) + 1
    }

    // --- Build the settings map ---
    def markerSeen = [:]
    def channelSettings = new LinkedHashMap()

    availableChannels.each { ch ->
        def marker  = extractMarker(ch.getName())
        def showing = selectedChannels.contains(ch)
        def minVal  = ch.getMinDisplay()
        def maxVal  = ch.getMaxDisplay()
        def hexColor = rgbToHex(ch.getColor())

        // If a marker appears more than once (e.g. HEM), number them
        def key = marker
        if (markerCounts[marker] > 1) {
            markerSeen[marker] = (markerSeen[marker] ?: 0) + 1
            key = "${marker}_${markerSeen[marker]}"
        }

        channelSettings[key] = [
            min:   Math.round(minVal  * 100.0) / 100.0,
            max:   Math.round(maxVal  * 100.0) / 100.0,
            show:  showing,
            color: hexColor
        ]
    }

    def preset = [source: imageName, channels: channelSettings]
    presetFile.text = gson.toJson(preset)

    println "=== CAPTURE COMPLETE ==="
    println "Source slide : ${imageName}"
    println "Preset name  : ${PRESET_NAME}"
    println "Saved to     : ${presetFile.name}"
    println ""
    println "Channels captured:"
    channelSettings.each { key, val ->
        def flag = val.show ? "ON " : "off"
        println "  [${flag}]  ${key}:  ${val.min} - ${val.max}  ${val.color}"
    }
    println ""
    println "Next steps:"
    println "  1. Open the slide you want to apply these settings to"
    println "  2. Change MODE to \"APPLY\" at the top of this script"
    println "  3. Run the script again"


} else if (MODE.toUpperCase() == "APPLY") {

    if (!presetFile.exists()) {
        println "ERROR: No preset named '${PRESET_NAME}' found."
        println "       Expected file: ${presetFile.absolutePath}"
        println "       Run this script in CAPTURE mode first."
        return
    }

    def mapType = new TypeToken<Map<String, Object>>(){}.getType()
    def preset = gson.fromJson(presetFile.text, mapType)
    println "=== APPLYING PRESET '${PRESET_NAME}' ==="
    println "Settings from : ${preset.source}"
    println "Applying to   : ${imageName}"
    println ""

    // --- Count duplicate markers on the TARGET slide ---
    def markerCounts = [:]
    availableChannels.each { ch ->
        def m = extractMarker(ch.getName())
        markerCounts[m] = (markerCounts[m] ?: 0) + 1
    }

    def markerSeen = [:]
    int applied = 0
    int skipped = 0

    availableChannels.each { ch ->
        def marker = extractMarker(ch.getName())

        def key = marker
        if (markerCounts[marker] > 1) {
            markerSeen[marker] = (markerSeen[marker] ?: 0) + 1
            key = "${marker}_${markerSeen[marker]}"
        }

        // Look up saved settings: try numbered key first, then base marker
        def s = preset.channels[key]
        if (s == null) {
            s = preset.channels[marker]
        }

        if (s != null) {
            imageDisplay.setMinMaxDisplay(ch, s.min as float, s.max as float)
            imageDisplay.setChannelSelected(ch, s.show as boolean)
            def colorStr = s.color ?: ""
            def flag = s.show ? "ON " : "off"
            println "  [${flag}]  ${ch.getName()}  <-  ${key}  [${s.min} - ${s.max}]  ${colorStr}"
            applied++
        } else {
            println "  [ ? ]  ${ch.getName()}  (${key})  -- no matching preset entry, skipped"
            skipped++
        }
    }

    // --- Apply channel colors via server metadata ---
    def imageData = getCurrentImageData()
    def serverMetadata = imageData.getServer().getMetadata()
    def oldChannelList = serverMetadata.getChannels()

    // Re-count markers for the metadata channel list
    def metaMarkerCounts = [:]
    oldChannelList.each { ch ->
        def m = extractMarker(ch.getName())
        metaMarkerCounts[m] = (metaMarkerCounts[m] ?: 0) + 1
    }

    def metaMarkerSeen = [:]
    boolean colorsChanged = false
    def newChannelList = oldChannelList.collect { ch ->
        def marker = extractMarker(ch.getName())
        def key = marker
        if (metaMarkerCounts[marker] > 1) {
            metaMarkerSeen[marker] = (metaMarkerSeen[marker] ?: 0) + 1
            key = "${marker}_${metaMarkerSeen[marker]}"
        }

        def s = preset.channels[key]
        if (s == null) s = preset.channels[marker]

        if (s != null && s.color != null) {
            def newColor = hexToPackedRGB(s.color as String)
            colorsChanged = true
            return ImageChannel.getInstance(ch.getName(), newColor)
        }
        return ch
    }

    if (colorsChanged) {
        def newMetadata = new ImageServerMetadata.Builder(serverMetadata)
            .channels(newChannelList)
            .build()
        imageData.updateServerMetadata(newMetadata)
        println "\nChannel colors updated."
    }

    // Force the viewer to redraw
    viewer.repaint()

    println ""
    println "Done!  ${applied} channels updated, ${skipped} skipped."
    if (skipped > 0) {
        println "(Skipped channels had no matching marker in the saved preset."
        println " This is normal when slides have different marker panels.)"
    }

} else if (MODE.toUpperCase() == "APPLY_ALL") {

    // -------------------------------------------------------
    //  APPLY_ALL: Generate a QuPath-native display preset
    //  containing entries for every channel name in the
    //  project. QuPath matches by channel name, so loading
    //  this preset on any slide applies the right settings.
    // -------------------------------------------------------

    if (!presetFile.exists()) {
        println "ERROR: No preset named '${PRESET_NAME}' found."
        println "       Expected file: ${presetFile.absolutePath}"
        println "       Run this script in CAPTURE mode first."
        return
    }

    // Read the captured marker-based settings
    def mapType = new TypeToken<Map<String, Object>>(){}.getType()
    def preset = gson.fromJson(presetFile.text, mapType)

    // Read the project file to get all images and channel names
    def projectFileContent = project.getPath().toFile().getText('UTF-8')
    def projectJson = gson.fromJson(projectFileContent, mapType)
    def images = projectJson.images as List

    println "=== APPLY_ALL ==="
    println "Preset '${PRESET_NAME}' captured from: ${preset.source}"
    println "Project contains ${images.size()} images"
    println ""

    // Build a single display preset with entries for every
    // channel name across all images
    def allChannelEntries = []
    def seenNames = new HashSet()
    int matched = 0
    int unmatched = 0

    for (img in images) {
        def imgName = img.serverBuilder?.metadata?.name ?: "unknown"
        def channels = img.serverBuilder?.metadata?.channels as List
        if (channels == null) continue

        // Count markers in this image for duplicate handling
        def markerCounts = [:]
        channels.each { ch ->
            def marker = extractMarker(ch.name as String)
            markerCounts[marker] = (markerCounts[marker] ?: 0) + 1
        }

        def markerSeen = [:]
        channels.each { ch ->
            def chName = ch.name as String

            // Skip if we already have an entry for this exact name
            if (seenNames.contains(chName)) return
            seenNames.add(chName)

            def marker = extractMarker(chName)
            def key = marker
            if (markerCounts[marker] > 1) {
                markerSeen[marker] = (markerSeen[marker] ?: 0) + 1
                key = "${marker}_${markerSeen[marker]}"
            }

            def s = preset.channels[key]
            if (s == null) s = preset.channels[marker]

            // Fallback color: unpack from project metadata
            def packed = (ch.color as Number).intValue()
            def fallbackColor = [
                red:   (packed >> 16) & 0xFF,
                green: (packed >> 8) & 0xFF,
                blue:  packed & 0xFF
            ]

            if (s != null) {
                // Use captured color if available, otherwise original
                def colorMap = (s.color != null) ? hexToRgbMap(s.color as String) : fallbackColor
                allChannelEntries.add([
                    name:       chName,
                    minDisplay: s.min,
                    maxDisplay: s.max,
                    color:      colorMap,
                    isShowing:  s.show
                ])
                matched++
            } else {
                // No saved setting for this marker -- include with
                // defaults so QuPath doesn't ignore the channel
                allChannelEntries.add([
                    name:       chName,
                    minDisplay: 0.0,
                    maxDisplay: 255.0,
                    color:      fallbackColor,
                    isShowing:  false
                ])
                unmatched++
            }
        }
    }

    // Write the QuPath-native display preset
    def displayPreset = new LinkedHashMap()
    displayPreset.put("name", "auto_${PRESET_NAME}")
    displayPreset.put("gamma", 1.0)
    displayPreset.put("invertBackground", false)
    displayPreset.put("channels", allChannelEntries)

    def displayDir = new File(projectDir, "resources/display")
    displayDir.mkdirs()
    def displayFile = new File(displayDir, "auto_${PRESET_NAME}.json")
    displayFile.setText(gson.toJson(displayPreset), 'UTF-8')

    println "Generated QuPath display preset:"
    println "  File     : resources/display/${displayFile.name}"
    println "  Channels : ${allChannelEntries.size()} entries (${matched} matched, ${unmatched} unmatched)"
    println "  Images   : ${images.size()}"
    println ""
    println "HOW TO USE:"
    println "  1. Open any slide in the project"
    println "  2. Open Brightness & Contrast  (View > Brightness & Contrast)"
    println "  3. Click the preset dropdown (top of the B&C panel)"
    println "  4. Select 'auto_${PRESET_NAME}'"
    println "  5. The display settings update instantly"
    println "  6. Repeat for each slide -- the same preset works on all of them"

} else {
    println "ERROR: MODE must be \"CAPTURE\", \"APPLY\", or \"APPLY_ALL\" (currently: \"${MODE}\")"
}
