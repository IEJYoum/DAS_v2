macro "AEC CMYK Color Deconvolution" {

	input = getArgument();

	if (input == "") {
		input = getDirectory("Select the folder of image sets");
	}

	suffix = ".tif";

	setBatchMode("true");
	pdir = getFileList(input);

	for (m = 1; m < pdir.length + 1; m++) {
		if (!startsWith(pdir[m-1], "Registration_Check/") && File.isDirectory(input + pdir[m-1])) {
			output = input + pdir[m-1] + "Processed";
			if (!File.exists(output)) {
				File.makeDirectory(output);
			}
		}
		if (endsWith(pdir[m-1], "Registered_Regions/"));
			rr = input + pdir[m-1] + "Registered_Regions/";
			rdir = getFileList(rr);
			roidirs = Array.sort(rdir);
			for (n = 1; n < rdir.length + 1; n++) {
				Rsave = input + pdir[m-1] + "Processed" + File.separator + roidirs[n-1];
				if (!File.exists(Rsave)); {
					File.makeDirectory(Rsave);
				}
				processRFolder(input + pdir[m-1] + "Registered_Regions/" + roidirs[n-1]);
			}
		}
}

function processRFolder(in) {
	list = getFileList(in);
	for (i = 1; i < list.length + 1; i++) {
		if (File.isDirectory(input + list[i-1]))
			processRFolder("" + input + list[i-1]);
		else {
			if (endsWith(list[i-1], suffix) || endsWith(list[i-1], ".tiff")) {
				if (!File.exists(Rsave + File.separator + "V_" + list[i-1])) {
					CD_cmyk(in, output, list[i-1]);
				}
			}
		}
	}
}

// -----------------------------------------------------------------------
// Replaces: run("RGB to CMYK")
// Delegates to an embedded Jython script for fast, array-level processing.
// Uses Java int[] pixel arrays directly — no per-pixel IJM overhead.
// Produces 8-bit images named "C", "M", "Y" in the ImageJ window list,
// matching the original plugin output consumed by CD_cmyk().
// -----------------------------------------------------------------------
function rgbToCMYK(imgID) {
	// Pass the image ID to the Jython script via the IJ property bus
	call("ij.Prefs.set", "cmyk.imgID", imgID);

	jythonScript =
		"from ij import IJ, ImagePlus\n" +
		"from ij.process import ByteProcessor\n" +
		"import jarray\n" +
		"\n" +
		"# Retrieve the image ID set by the macro\n" +
		"imgID = int(IJ.runMacro('call(\"ij.Prefs.get\", \"cmyk.imgID\", \"0\")'))\n" +
		"imp = IJ.getImage() if imgID == 0 else [x for x in map(lambda i: IJ.getImage(), [0]) if True][0]\n" +
		"# Robust: get by ID\n" +
		"from ij import WindowManager\n" +
		"imp = WindowManager.getImage(imgID)\n" +
		"\n" +
		"ip  = imp.getProcessor()\n" +
		"w   = imp.getWidth()\n" +
		"h   = imp.getHeight()\n" +
		"n   = w * h\n" +
		"\n" +
		"# Get packed RGB pixel array (Java int[])\n" +
		"pixels = ip.getPixels()\n" +
		"\n" +
		"# Allocate output byte arrays\n" +
		"cArr = jarray.zeros(n, 'b')\n" +
		"mArr = jarray.zeros(n, 'b')\n" +
		"yArr = jarray.zeros(n, 'b')\n" +
		"\n" +
		"INV255 = 1.0 / 255.0\n" +
		"\n" +
		"for i in xrange(n):\n" +
		"    v  = pixels[i] & 0xFFFFFFFF\n" +
		"    r  = ((v >> 16) & 0xFF) * INV255\n" +
		"    g  = ((v >>  8) & 0xFF) * INV255\n" +
		"    b  =  (v        & 0xFF) * INV255\n" +
		"    mx = max(r, g, b)\n" +
		"    if mx == 0.0:\n" +
		"        cArr[i] = 0; mArr[i] = 0; yArr[i] = 0\n" +
		"    else:\n" +
		"        k = 1.0 - mx\n" +
		"        cv = int(min(255, max(0, round((1.0 - r - k) / mx * 255))))\n" +
		"        mv = int(min(255, max(0, round((1.0 - g - k) / mx * 255))))\n" +
		"        yv = int(min(255, max(0, round((1.0 - b - k) / mx * 255))))\n" +
		"        # Store as signed byte (Java byte is signed)\n" +
		"        cArr[i] = cv if cv < 128 else cv - 256\n" +
		"        mArr[i] = mv if mv < 128 else mv - 256\n" +
		"        yArr[i] = yv if yv < 128 else yv - 256\n" +
		"\n" +
		"# Build ImagePlus objects and display with expected names\n" +
		"cImp = ImagePlus('C', ByteProcessor(w, h, cArr, None))\n" +
		"mImp = ImagePlus('M', ByteProcessor(w, h, mArr, None))\n" +
		"yImp = ImagePlus('Y', ByteProcessor(w, h, yArr, None))\n" +
		"cImp.show()\n" +
		"mImp.show()\n" +
		"yImp.show()\n";

	eval("script", jythonScript);
}

function CD_cmyk(in, output, filename) {
	if (!startsWith(filename, "NUCLEI_") == 1) {
		open(in + filename);
		markerID = getImageID();
		marker = getTitle();

		// ---- Inline RGB->CMYK (replaces run("RGB to CMYK")) ----
		rgbToCMYK(markerID);
		// After rgbToCMYK, open images named "C", "M", "Y" exist as 8-bit

		// Add Y + M (same as original: imageCalculator("Add create", "Y","M"))
		imageCalculator("Add create", "Y", "M");
		mimg = getImageID();

		close("M");
		close("K");  // K is not created by our function but kept for safety
		close("C");
		close("Y");

		selectImage(mimg);
		run("Grays");
		run("8-bit");
		getRawStatistics(nPixels, mean, min, max);
		newmin = max * 0.05;
		newmax = max * 0.95;
		setMinAndMax(newmin, newmax);
		run("Apply LUT");
		saveAs("Tiff", Rsave + File.separator + "V_" + marker);
		print(filename + " saved");
		close();
		close(marker);
	}
}