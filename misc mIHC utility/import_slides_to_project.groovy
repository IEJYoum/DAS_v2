/*
 * import_slides_to_project.groovy
 *
 * Batch-imports all slide images from a parent folder into the currently open
 * QuPath project.
 *
 * Expected layout:
 *   SLIDES_ROOT/
 *     SlideA/
 *       image1.svs
 *       image2.svs
 *     SlideB/
 *       image1.svs
 *       ...
 *
 * Usage:
 *   1. Create or open a QuPath project (can be empty).
 *   2. Set SLIDES_ROOT below.
 *   3. Run this script from Automate > Script Editor.
 *
 * Supported formats: .svs, .tif, .tiff, .ndpi, .qptiff, .mrxs, .vsi, .scn
 * Add more to IMAGE_EXTENSIONS if needed.
 */

// ===================== USER SETTINGS =====================

def SLIDES_ROOT = "Z:/Multiplex_IHC_studies/UCSD_AndyLowey_Lustgarten/OHSU_mIHC_UCSD_Lowey/Slides"

// Set to a specific slide folder name (e.g. "FCPDAC002") to import only that
// slide, or leave empty to import everything.
def SLIDE_FILTER = ""

// If true, skip files that are already in the project (matched by URI).
def SKIP_EXISTING = true

// =================== END USER SETTINGS ===================

import qupath.lib.projects.ProjectIO
import qupath.lib.images.servers.ImageServerProvider
import java.awt.image.BufferedImage

def IMAGE_EXTENSIONS = [".svs", ".tif", ".tiff", ".ndpi", ".qptiff", ".mrxs", ".vsi", ".scn"] as Set

def project = getProject()
if (project == null) {
    println "ERROR: No project is open. Create or open a QuPath project first."
    return
}

def root = new File(SLIDES_ROOT)
if (!root.isDirectory()) {
    println "ERROR: SLIDES_ROOT is not a valid directory: ${SLIDES_ROOT}"
    return
}

// Collect existing image URIs so we can skip duplicates.
def existingURIs = new HashSet<String>()
if (SKIP_EXISTING) {
    for (entry in project.getImageList()) {
        try {
            def uris = entry.getURIs()
            for (uri in uris) {
                existingURIs.add(uri.toString())
            }
        } catch (Exception e) {
            // entry may not have readable URIs yet
        }
    }
    if (existingURIs.size() > 0) {
        println "Project already contains ${existingURIs.size()} image URI(s); will skip duplicates."
    }
}

def slideFolders = root.listFiles()
    .findAll { it.isDirectory() }
    .sort { it.name }

if (SLIDE_FILTER != null && SLIDE_FILTER.trim() != "") {
    slideFolders = slideFolders.findAll { it.name == SLIDE_FILTER.trim() }
}

println "Found ${slideFolders.size()} slide folder(s) under ${root.name}"

int added = 0
int skipped = 0
int failed = 0

for (slideFolder in slideFolders) {
    def imageFiles = slideFolder.listFiles()
        .findAll { f ->
            f.isFile() && IMAGE_EXTENSIONS.any { ext -> f.name.toLowerCase().endsWith(ext) }
        }
        .sort { it.name }

    if (imageFiles.isEmpty()) {
        println "  ${slideFolder.name}: no image files found, skipping."
        continue
    }

    println "  ${slideFolder.name}: ${imageFiles.size()} image(s)"

    for (imageFile in imageFiles) {
        try {
            def uri = imageFile.toURI()
            if (SKIP_EXISTING && existingURIs.contains(uri.toString())) {
                skipped++
                continue
            }

            def support = ImageServerProvider.getPreferredUriImageSupport(
                BufferedImage.class, uri.toString()
            )
            if (support == null) {
                println "    SKIP (no reader): ${imageFile.name}"
                skipped++
                continue
            }

            def builders = support.getBuilders()
            if (builders == null || builders.isEmpty()) {
                println "    SKIP (no builders): ${imageFile.name}"
                skipped++
                continue
            }

            for (builder in builders) {
                def entry = project.addImage(builder)
                if (entry != null) {
                    // Set the image name to something readable.
                    entry.setImageName(imageFile.name)
                    added++
                }
            }
        } catch (Exception e) {
            println "    FAILED: ${imageFile.name} -> ${e.message}"
            failed++
        }
    }
}

// Persist the project file.
project.syncChanges()

println ""
println "Done. Added: ${added}, Skipped: ${skipped}, Failed: ${failed}"
println "Total project entries: ${project.getImageList().size()}"
