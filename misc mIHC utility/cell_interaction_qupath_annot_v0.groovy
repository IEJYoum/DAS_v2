/*
 * Cell interaction hotspot annotator, v0 (image-only)
 *
 * Run in QuPath with one image open.  It reads the full image at a chosen
 * downsample, builds corrected/smoothed marker maps, scores their joint spatial
 * pattern, and creates one annotation for each accepted peak.
 *
 * Designed for QuPath 0.6+.  This is a first-pass candidate-niche finder;
 * inspect score/marker overlays and tune on representative slides before
 * applying a project-wide rule.
 */

import static qupath.lib.gui.scripting.QPEx.*

import ij.plugin.filter.GaussianBlur
import ij.process.FloatProcessor
import org.bytedeco.opencv.opencv_core.Mat
import qupath.lib.objects.PathObjects
import qupath.lib.objects.classes.PathClass
import qupath.lib.regions.RegionRequest
import qupath.opencv.tools.OpenCVTools

import java.util.ArrayList
import java.util.Arrays
import java.util.Comparator
import java.util.PriorityQueue


// -----------------------------------------------------------------------------
// GLOBAL KNOBS - edit these for the interaction you want to find
// -----------------------------------------------------------------------------

// Every positive marker must be present.  A context marker (e.g. PanCK for an
// in-tumor query) is simply another positive marker; do not require a separate
// compartment annotation.
//
// floor/ceiling are raw image values.  Values <= floor become 0; values >=
// ceiling become 1.  Choose them after inspecting each channel's intensity range.
// gate is applied AFTER smoothing, on the normalized 0-1 map.
def POSITIVE_MARKERS = [
    [name: 'CD8',  floor: 100.0, ceiling: 3000.0, weight: 1.0, gate: 0.08],
    [name: 'FAP',  floor: 100.0, ceiling: 3000.0, weight: 1.0, gate: 0.08]
    // Example tumor-context gate:
    // [name: 'PanCK', floor: 100.0, ceiling: 3000.0, weight: 1.0, gate: 0.05]
]

// Negative markers penalize the score only above their floor.  This is the crude
// background subtraction that stops low CD4 background from determining a result.
// penalty controls how strongly a normalized negative signal suppresses the score.
def NEGATIVE_MARKERS = [
    [name: 'CD4', floor: 100.0, ceiling: 3000.0, weight: 1.0, penalty: 2.0]
]

// Working resolution and smoothing scale.  Both are physical units, not pixels.
double WORKING_PIXEL_SIZE_UM = 4.0
double BLUR_SIGMA_UM = 40.0
int MAX_WORKING_PIXELS = 4_000_000            // safety limit; raises downsample if needed

// Peak and annotation rules
int N_HOTSPOTS = 2
double PEAK_SCORE_FLOOR = 0.03
double MIN_COMPONENT_SCORE = 0.005
int THRESHOLD_STEPS = 80
double MIN_PEAK_SEPARATION_UM = 150.0
double TARGET_AREA_UM2 = 1_000.0
double MAX_AREA_UM2 = 20_000.0
int CONNECTIVITY = 8                           // 4 or 8
int MAX_CANDIDATE_PEAKS = 20_000               // keeps memory bounded on large slides

// Optional mask only.  If true, select a single annotation before running; this
// restricts processing to that annotation's ROI.  It is NOT required for a
// PanCK/tumor-style context query.
boolean USE_SELECTED_ANNOTATION_AS_MASK = false

// Output behavior
String OUTPUT_CLASS_NAME = 'Cell interaction hotspot'
String OUTPUT_NAME_PREFIX = 'Interaction hotspot'
boolean DELETE_PREVIOUS_OUTPUT_WITH_SAME_PREFIX = false


// -----------------------------------------------------------------------------
// Validation and image setup
// -----------------------------------------------------------------------------

if (POSITIVE_MARKERS.isEmpty())
    throw new IllegalArgumentException('Specify at least one positive marker')
if (CONNECTIVITY != 4 && CONNECTIVITY != 8)
    throw new IllegalArgumentException('CONNECTIVITY must be 4 or 8')
if (TARGET_AREA_UM2 <= 0 || MAX_AREA_UM2 < TARGET_AREA_UM2)
    throw new IllegalArgumentException('Use 0 < TARGET_AREA_UM2 <= MAX_AREA_UM2')
if (THRESHOLD_STEPS < 1 || MIN_COMPONENT_SCORE > PEAK_SCORE_FLOOR)
    throw new IllegalArgumentException('Check threshold settings')

def imageData = getCurrentImageData()
if (imageData == null)
    throw new IllegalStateException('Open an image in QuPath before running this script')

def server = imageData.getServer()
def calibration = server.getPixelCalibration()
if (!calibration.hasPixelSizeMicrons())
    throw new IllegalStateException('Image calibration in microns is required')

double basePixelSizeUm = calibration.getAveragedPixelSizeMicrons()
double requestedDownsample = Math.max(1.0, WORKING_PIXEL_SIZE_UM / basePixelSizeUm)
double fullResolutionPixels = (double)server.getWidth() * (double)server.getHeight()
double minimumDownsampleForMemory = Math.sqrt(fullResolutionPixels / MAX_WORKING_PIXELS)
double downsample = Math.max(requestedDownsample, minimumDownsampleForMemory)

def request = RegionRequest.createInstance(server, downsample)
def image = server.readRegion(request)
int width = image.getWidth()
int height = image.getHeight()
int nPixels = width * height
double workingPixelSizeUm = basePixelSizeUm * downsample
double workingPixelAreaUm2 = workingPixelSizeUm * workingPixelSizeUm

def metadataChannels = server.getMetadata().getChannels()
def availableChannelNames = metadataChannels.collect { it.getName() }
int nBands = image.getRaster().getNumBands()

def resolveChannelIndex = { String channelName ->
    int index = availableChannelNames.findIndexOf { it.equalsIgnoreCase(channelName) }
    if (index < 0)
        throw new IllegalArgumentException("Channel '${channelName}' not found. Available: ${availableChannelNames}")
    if (index >= nBands)
        throw new IllegalArgumentException("Channel '${channelName}' resolved to index ${index}, but the requested image has only ${nBands} bands")
    return index
}

// Do not silently combine duplicate marker names with different settings.
def allMarkerNames = (POSITIVE_MARKERS + NEGATIVE_MARKERS).collect { it.name as String }
if (allMarkerNames.size() != allMarkerNames.toSet().size())
    throw new IllegalArgumentException('A marker may appear only once across POSITIVE_MARKERS and NEGATIVE_MARKERS')


// -----------------------------------------------------------------------------
// Optional selected-annotation mask
// -----------------------------------------------------------------------------

boolean[] analysisMask = new boolean[nPixels]
Arrays.fill(analysisMask, true)
if (USE_SELECTED_ANNOTATION_AS_MASK) {
    def selected = getSelectedObject()
    if (selected == null || !selected.hasROI() || !selected.getROI().isArea())
        throw new IllegalArgumentException('Select one area annotation, or set USE_SELECTED_ANNOTATION_AS_MASK = false')
    def selectedROI = selected.getROI()
    for (int y = 0; y < height; y++) {
        double fullY = (y + 0.5) * downsample
        int row = y * width
        for (int x = 0; x < width; x++)
            analysisMask[row + x] = selectedROI.contains((x + 0.5) * downsample, fullY)
    }
}


// -----------------------------------------------------------------------------
// Correct and blur each marker map
// -----------------------------------------------------------------------------

def raster = image.getRaster()
double sigmaWorkingPixels = BLUR_SIGMA_UM / workingPixelSizeUm
def smoothedMaps = [:]

def buildCorrectedSmoothedMap = { Map marker ->
    if (marker.ceiling <= marker.floor)
        throw new IllegalArgumentException("${marker.name}: ceiling must exceed floor")

    int channelIndex = resolveChannelIndex(marker.name as String)
    float[] values = new float[nPixels]
    double denominator = marker.ceiling - marker.floor

    for (int y = 0; y < height; y++) {
        int row = y * width
        for (int x = 0; x < width; x++) {
            int i = row + x
            if (!analysisMask[i])
                continue
            double raw = raster.getSampleDouble(x, y, channelIndex)
            values[i] = (float)Math.max(0.0, Math.min(1.0, (raw - marker.floor) / denominator))
        }
    }

    def processor = new FloatProcessor(width, height, values)
    if (sigmaWorkingPixels > 0)
        new GaussianBlur().blurGaussian(processor, sigmaWorkingPixels, sigmaWorkingPixels, 0.01)
    return (float[])processor.getPixels()
}

(POSITIVE_MARKERS + NEGATIVE_MARKERS).each { marker ->
    smoothedMaps[marker.name] = buildCorrectedSmoothedMap(marker)
}


// -----------------------------------------------------------------------------
// Score map: product of positives / penalty from negatives
// -----------------------------------------------------------------------------

float[] score = new float[nPixels]
for (int i = 0; i < nPixels; i++) {
    if (!analysisMask[i])
        continue

    boolean passesPositiveGates = true
    double s = 1.0
    for (marker in POSITIVE_MARKERS) {
        double value = smoothedMaps[marker.name][i]
        if (value < marker.gate) {
            passesPositiveGates = false
            break
        }
        s *= Math.pow(Math.max(value, 1e-8), marker.weight)
    }
    if (!passesPositiveGates)
        continue

    for (marker in NEGATIVE_MARKERS) {
        double value = smoothedMaps[marker.name][i]
        s /= Math.pow(1.0 + marker.penalty * value, marker.weight)
    }
    score[i] = (float)s
}


// -----------------------------------------------------------------------------
// Find distinct local maxima, then grow one connected component around each peak
// -----------------------------------------------------------------------------

def scoreComparator = { Integer a, Integer b -> Float.compare(score[a], score[b]) } as Comparator<Integer>
def localPeaks = new PriorityQueue<Integer>(MAX_CANDIDATE_PEAKS, scoreComparator)

for (int y = 1; y < height - 1; y++) {
    int row = y * width
    for (int x = 1; x < width - 1; x++) {
        int i = row + x
        float value = score[i]
        if (value < PEAK_SCORE_FLOOR)
            continue

        boolean isPeak = true
        for (int dy = -1; dy <= 1 && isPeak; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0)
                    continue
                if (score[(y + dy) * width + x + dx] > value) {
                    isPeak = false
                    break
                }
            }
        }
        if (!isPeak)
            continue

        if (localPeaks.size() < MAX_CANDIDATE_PEAKS)
            localPeaks.add(i)
        else if (value > score[localPeaks.peek()]) {
            localPeaks.poll()
            localPeaks.add(i)
        }
    }
}

def sortedPeaks = new ArrayList<Integer>(localPeaks)
sortedPeaks.sort { Integer a, Integer b -> Float.compare(score[b], score[a]) }

int[] labels = new int[nPixels]       // zero = unassigned; output labels start at one
int[] visited = new int[nPixels]      // integer stamps avoid clearing a large array
int[] queue = new int[nPixels]
int visitStamp = 0
int nextLabel = 1
def accepted = []
double minSeparationPixelsSquared = Math.pow(MIN_PEAK_SEPARATION_UM / workingPixelSizeUm, 2)

def growComponent = { int seed, double threshold, int stamp ->
    int head = 0
    int tail = 0
    queue[tail++] = seed
    visited[seed] = stamp

    while (head < tail) {
        int current = queue[head++]
        int x = current % width
        int y = (int)(current / width)
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0)
                    continue
                if (CONNECTIVITY == 4 && Math.abs(dx) + Math.abs(dy) != 1)
                    continue
                int nx = x + dx
                int ny = y + dy
                if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                    continue
                int neighbor = ny * width + nx
                if (visited[neighbor] == stamp || labels[neighbor] != 0)
                    continue
                if (score[neighbor] < threshold)
                    continue
                visited[neighbor] = stamp
                queue[tail++] = neighbor
            }
        }
    }
    return tail
}

for (int seed : sortedPeaks) {
    if (accepted.size() >= N_HOTSPOTS)
        break
    if (labels[seed] != 0)
        continue

    int seedX = seed % width
    int seedY = (int)(seed / width)
    boolean tooClose = accepted.any { result ->
        double dx = seedX - result.seedX
        double dy = seedY - result.seedY
        return dx * dx + dy * dy < minSeparationPixelsSquared
    }
    if (tooClose)
        continue

    int selectedSize = 0
    double selectedThreshold = Double.NaN
    for (int step = 0; step <= THRESHOLD_STEPS; step++) {
        double fraction = step / (double)THRESHOLD_STEPS
        double threshold = score[seed] - fraction * (score[seed] - MIN_COMPONENT_SCORE)
        visitStamp++
        int componentSize = growComponent(seed, threshold, visitStamp)
        double componentArea = componentSize * workingPixelAreaUm2

        if (componentArea > MAX_AREA_UM2)
            break
        if (componentArea >= TARGET_AREA_UM2) {
            selectedSize = componentSize
            selectedThreshold = threshold
            break
        }
    }
    if (selectedSize == 0)
        continue

    // queue currently contains the first component that reached target area.
    for (int q = 0; q < selectedSize; q++)
        labels[queue[q]] = nextLabel
    accepted << [label: nextLabel, seedX: seedX, seedY: seedY,
                 peakScore: score[seed], threshold: selectedThreshold,
                 rasterAreaUm2: selectedSize * workingPixelAreaUm2]
    nextLabel++
}


// -----------------------------------------------------------------------------
// Turn accepted labelled components into QuPath annotations and measurements
// -----------------------------------------------------------------------------

def hierarchy = imageData.getHierarchy()
def outputClass = PathClass.fromString(OUTPUT_CLASS_NAME)

if (DELETE_PREVIOUS_OUTPUT_WITH_SAME_PREFIX) {
    def oldObjects = getAnnotationObjects().findAll {
        it.getName() != null && it.getName().startsWith(OUTPUT_NAME_PREFIX) && it.getPathClass() == outputClass
    }
    if (!oldObjects.isEmpty())
        hierarchy.removeObjects(oldObjects, true)
}

if (!accepted.isEmpty()) {
    def matLabels = new Mat(height, width, org.bytedeco.opencv.global.opencv_core.CV_32SC1)
    matLabels.put(0, 0, labels)
    def roisByLabel = OpenCVTools.createROIs(matLabels, request, 1, -1)
    def resultsByLabel = accepted.collectEntries { [(it.label as Number): it] }
    def newAnnotations = []

    roisByLabel.each { label, roi ->
        def result = resultsByLabel[label]
        if (result == null)
            return
        def annotation = PathObjects.createAnnotationObject(roi, outputClass)
        annotation.setName(String.format('%s %02d', OUTPUT_NAME_PREFIX, result.label))
        def measurements = annotation.getMeasurementList()
        measurements.put('Interaction peak score', result.peakScore)
        measurements.put('Interaction component threshold', result.threshold)
        measurements.put('Interaction raster area um^2', result.rasterAreaUm2)
        measurements.put('Interaction working pixel size um', workingPixelSizeUm)
        measurements.put('Interaction blur sigma um', BLUR_SIGMA_UM)
        measurements.put('Interaction seed x (working px)', result.seedX)
        measurements.put('Interaction seed y (working px)', result.seedY)
        newAnnotations << annotation
    }
    hierarchy.addObjects(newAnnotations)
    matLabels.close()
}

fireHierarchyUpdate()
println "Cell interaction v0 complete: accepted ${accepted.size()} of ${N_HOTSPOTS} requested hotspots."
println "Working image: ${width} x ${height} px; working pixel size: ${String.format('%.3f', workingPixelSizeUm)} um; sigma: ${BLUR_SIGMA_UM} um."
if (accepted.isEmpty())
    println 'No hotspot reached TARGET_AREA_UM2 before MIN_COMPONENT_SCORE or MAX_AREA_UM2. Adjust marker floors/gates, score floors, blur, or area knobs.'
