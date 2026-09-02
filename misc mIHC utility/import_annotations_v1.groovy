/*
 * import_annotations_v1.groovy
 *
 * Imports GeoJSON annotations produced by cell_interaction_v1.py into the
 * currently open QuPath image.  Run from QuPath's script editor with one
 * image open.
 *
 * NOTE: QuPath 0.7+ has a built-in menu for this:
 *       File > Import objects from file
 *   That is the safest approach. This script exists as an alternative
 *   for batch workflows or automation.
 *
 * This script only ADDS annotations.  It does not modify image data,
 * channels, or project structure.  Delete the annotations or don't save
 * if the result is not wanted.
 */

import javafx.stage.FileChooser
import qupath.fx.dialogs.FileChoosers
import qupath.lib.io.PathIO

def filter = new FileChooser.ExtensionFilter("GeoJSON files", "*.geojson")
def geojsonFile = FileChoosers.promptForFile("Select annotations.geojson", filter)
if (geojsonFile == null) {
    println "Cancelled -- no file selected."
    return
}

def annotations = PathIO.readObjects(geojsonFile)
if (annotations.isEmpty()) {
    println "No annotation objects found in ${geojsonFile.name}."
    return
}

getCurrentHierarchy().addObjects(annotations)
fireHierarchyUpdate()
println "Imported ${annotations.size()} annotations from ${geojsonFile.name}"
