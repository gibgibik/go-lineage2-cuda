package main

import (
	json2 "encoding/json"
	"fmt"
	"gocv.io/x/gocv"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"log"
	"math"
	"net/http"
	"sort"
)

var (
	excludeBoundsArea = []image.Rectangle{
		image.Rect(0, 0, 247, 104),
		image.Rect(0, 590, 370, 1074),
		image.Rect(697, 915, 1273, 1074),
		image.Rect(1710, 0, 1920, 350),
		image.Rect(1644, 0, 1748, 35),
		image.Rect(902, 478, 1040, 649),
	}
	npcThreshold = 0.9995
	npcNms       = 0.4
	resizeWidth  = 1920
	resizeHeight = 1088
	rW           = float64(1920) / float64(resizeWidth)
	rH           = float64(1080) / float64(resizeHeight)
)

type BoxesStruct struct {
	Boxes [][]int `json:"boxes"`
}

func findBoundsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST allowed", http.StatusMethodNotAllowed)
		return
	}

	// читаємо тіло запиту
	cpImg, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()
	mat, _ := gocv.IMDecode(cpImg, gocv.IMReadColor)
	blob := gocv.BlobFromImage(mat, 1.0, image.Pt(int(resizeWidth), int(resizeHeight)), gocv.NewScalar(123.68, 116.78, 103.94, 0), true, false) //frozen_east
	net := gocv.ReadNet("frozen_east_text_detection.pb", "")
	defer net.Close()

	net.SetPreferableBackend(gocv.NetBackendCUDA)
	net.SetPreferableTarget(gocv.NetTargetCUDA)
	if net.Empty() {
		log.Fatal("Failed to load the model")
	}

	//_ = net.SetPreferableBackend(gocv.NetBackendCUDA)
	//_ = net.SetPreferableTarget(gocv.NetTargetCUDA)
	net.SetInput(blob, "")
	// Define output layers
	outputNames := []string{"feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"}
	outputBlobs := net.ForwardLayers(outputNames)

	//
	//Decode results (this part can be tricky, depends on your task)
	scores := outputBlobs[0]
	geometry := outputBlobs[1]
	//scores := net.Forward("feature_fusion/Conv_7/Sigmoid")
	//geometry := net.Forward("feature_fusion/concat_3")
	fmt.Println("scores size:", scores.Size())
	fmt.Println("geometry size:", geometry.Size())
	rotatedBoxes, confidences := decodeBoundingBoxes(scores, geometry, float32(npcThreshold))
	boxes := []image.Rectangle{}
	for _, rotatedBox := range rotatedBoxes {
		//if !checkExcludeBox(rotatedBox.BoundingRect) {
		//	continue
		//}
		boxes = append(boxes, rotatedBox.BoundingRect)
	}
	// Only Apply NMS when there are at least one box
	indices := make([]int, len(boxes))
	if len(boxes) > 0 {
		indices = gocv.NMSBoxes(boxes, confidences, float32(npcThreshold), float32(npcNms))
	}
	// Resize indices to only include those that have values other than zero
	var numIndices int = 0
	for _, value := range indices {
		if value != 0 {
			numIndices++
		}
	}
	indices = indices[0:numIndices]
	var result BoxesStruct
	finalResult := BoxesStruct{
		Boxes: make([][]int, 0),
	}
	for i := 0; i < len(indices); i++ {
		// get 4 corners of the rotated rect
		verticesMat := gocv.NewMat()
		if err := gocv.BoxPoints(rotatedBoxes[indices[i]], &verticesMat); err != nil {
			log.Fatal(err)
		}

		//
		//	// scale the bounding box coordinates based on the respective ratios
		vertices := []image.Point{}
		var minX, minY, maxX, maxY int
		for j := 0; j < 4; j++ {
			p1 := image.Pt(
				int(verticesMat.GetFloatAt(j, 0)*float32(rW)),
				int(verticesMat.GetFloatAt(j, 1)*float32(rH)),
			)

			//p2 := image.Pt(
			//	int(verticesMat.GetFloatAt((j+1)%4, 0)*float32(rW)),
			//	int(verticesMat.GetFloatAt((j+1)%4, 1)*float32(rH)),
			//)
			if minX == 0 || minX > p1.X {
				minX = p1.X
			}
			if minY == 0 || minY > p1.Y {
				minY = p1.Y
			}
			if maxX == 0 || maxX < p1.X {
				maxX = p1.X
			}
			if maxY == 0 || maxY < p1.Y {
				maxY = p1.Y
			}
			vertices = append(vertices, p1)
			//gocv.Line(&mat, p1, p2, color.RGBA{0, 255, 0, 0}, 1)
		}
		rect := image.Rect(minX, minY, maxX, maxY)
		if !checkExcludeBox(rect) {
			continue
		}
		//fmt.Println(whitePixelPercentage(mat, rect))

		if whitePixelPercentage(mat, rect) < 10 {
			continue
		}

		result = BoxesStruct{
			Boxes: append(result.Boxes, []int{rect.Min.X, rect.Min.Y, rect.Max.X, rect.Max.Y}),
		}
		//continue //@todo remove
		//cropped := fourPointsTransform(mat, gocv.NewPointVectorFromPoints(vertices))
		//// Create a 4D blob from cropped image
		////blob = gocv.BlobFromImage(cropped, 1/127.5, image.Pt(128, 32), gocv.NewScalar(127.5, 0, 0, 0), false, false) //120?
		//buf, _ := gocv.IMEncode(gocv.PNGFileExt, cropped)
		//npcClient.SetImageFromBytes(buf.GetBytes())
		//foundText, _ := npcClient.Text()
		//
		//if _, ok := internal.NpcList[foundText]; ok {
		//gocv.Rectangle(&mat, rect, color.RGBA{0, 255, 0, 0}, 1)

		//_ = gocv.IeMWrite("output.png", mat)

		//fmt.Println(1)
		//}

	}

	for _, v := range groupAndSortRects(result.Boxes) {
		finalResult.Boxes = append(finalResult.Boxes, mergeCloseRectsInLine(v)...)
	}

	//for _, r := range finalResult.Boxes {
	//	go Draw(Hwnd, uintptr(r[0]), uintptr(r[1]), uintptr(r[2]), uintptr(r[3]), "")
	//
	//}
	_ = blob.Close()
	b, _ := json2.Marshal(finalResult)
	w.Write(b)
}

func main() {
	http.HandleFunc("/findBounds", findBoundsHandler)
	fmt.Println("Server started at :2224")
	log.Fatal(http.ListenAndServe(":2224", nil))
}

func decodeBoundingBoxes(scores gocv.Mat, geometry gocv.Mat, threshold float32) (detections []gocv.RotatedRect, confidences []float32) {
	scoresChannel := gocv.GetBlobChannel(scores, 0, 0)
	x0DataChannel := gocv.GetBlobChannel(geometry, 0, 0)
	x1DataChannel := gocv.GetBlobChannel(geometry, 0, 1)
	x2DataChannel := gocv.GetBlobChannel(geometry, 0, 2)
	x3DataChannel := gocv.GetBlobChannel(geometry, 0, 3)
	angleChannel := gocv.GetBlobChannel(geometry, 0, 4)

	for y := 0; y < scoresChannel.Rows(); y++ {
		for x := 0; x < scoresChannel.Cols(); x++ {

			// Extract data from scores
			score := scoresChannel.GetFloatAt(y, x)

			// If score is lower than threshold score, move to next x
			if score < threshold {
				continue
			}

			x0Data := x0DataChannel.GetFloatAt(y, x)
			x1Data := x1DataChannel.GetFloatAt(y, x)
			x2Data := x2DataChannel.GetFloatAt(y, x)
			x3Data := x3DataChannel.GetFloatAt(y, x)
			angle := angleChannel.GetFloatAt(y, x)

			// Calculate offset
			// Multiple by 4 because feature maps are 4 time less than input image.
			offsetX := x * 4.0
			offsetY := y * 4.0

			// Calculate cos and sin of angle
			cosA := math.Cos(float64(angle))
			sinA := math.Sin(float64(angle))

			h := x0Data + x2Data
			w := x1Data + x3Data

			// Calculate offset
			offset := []float64{
				(float64(offsetX) + cosA*float64(x1Data) + sinA*float64(x2Data)),
				(float64(offsetY) - sinA*float64(x1Data) + cosA*float64(x2Data)),
			}

			// Find points for rectangle
			p1 := []float64{
				(-sinA*float64(h) + offset[0]),
				(-cosA*float64(h) + offset[1]),
			}
			p3 := []float64{
				(-cosA*float64(w) + offset[0]),
				(sinA*float64(w) + offset[1]),
			}

			center := image.Pt(
				int(0.5*(p1[0]+p3[0])),
				int(0.5*(p1[1]+p3[1])),
			)

			detections = append(detections, gocv.RotatedRect{
				Points: []image.Point{
					{int(p1[0]), int(p1[1])},
					{int(p3[0]), int(p3[1])},
				},
				BoundingRect: image.Rect(
					int(p1[0]), int(p1[1]),
					int(p3[0]), int(p3[1]),
				),
				Center: center,
				Width:  int(w),
				Height: int(h),
				Angle:  float64(-1 * angle * 180 / math.Pi),
			})
			confidences = append(confidences, score)
		}
	}

	// Return detections and confidences
	return
}

func checkExcludeBox(box image.Rectangle) bool {
	for _, excludeBox := range excludeBoundsArea {
		if excludeBox.Min.X <= box.Min.X && excludeBox.Min.Y <= box.Min.Y && excludeBox.Max.X >= box.Max.X && excludeBox.Max.Y >= box.Max.Y {
			return false
		}
	}
	return true
}

func whitePixelPercentage(mat gocv.Mat, rect image.Rectangle) float64 {
	threshold := 80
	if rect.Min.X < 0 || rect.Min.Y < 0 || rect.Max.X > mat.Cols() || rect.Max.Y > mat.Rows() {
		return 0
	}

	roi := mat.Region(rect)
	defer roi.Close()

	// Convert to grayscale
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(roi, &gray, gocv.ColorBGRToGray)

	// Create binary mask where brightness >= threshold
	mask := gocv.NewMat()
	defer mask.Close()
	gocv.Threshold(gray, &mask, float32(threshold), 255, gocv.ThresholdBinary)

	// Debug: Save to verify
	gocv.IMWrite("debug_gray.jpg", gray)
	gocv.IMWrite("debug_white_mask.jpg", mask)

	// Calculate percent
	totalPixels := gray.Rows() * gray.Cols()
	if totalPixels == 0 {
		return 0
	}
	whitePixels := gocv.CountNonZero(mask)
	return float64(whitePixels) / float64(totalPixels) * 100.0
}

func groupAndSortRects(rects [][]int) [][][]int {
	yTolerance := 3
	var groups [][][]int

	for _, rect := range rects {
		y := rect[1] // Y1
		placed := false

		for i := range groups {
			gy := groups[i][0][1]
			if abs(gy-y) <= yTolerance {
				groups[i] = append(groups[i], rect)
				placed = true
				break
			}
		}
		if !placed {
			groups = append(groups, [][]int{rect})
		}
	}

	// Сортування по X1 всередині кожної групи
	for i := range groups {
		sort.Slice(groups[i], func(a, b int) bool {
			return groups[i][a][0] < groups[i][b][0] // X1
		})
	}

	// Сортування груп по Y1
	sort.Slice(groups, func(a, b int) bool {
		return groups[a][0][1] < groups[b][0][1] // Y1
	})

	return groups
}

func mergeCloseRectsInLine(line [][]int) [][]int {
	xTolerance := 10
	if len(line) == 0 {
		return nil
	}

	merged := [][]int{}
	current := line[0]

	for i := 1; i < len(line); i++ {
		next := line[i]

		// Якщо наступний дуже близько — об’єднуємо
		if next[0]-current[2] <= xTolerance {
			// об’єднати current і next
			current = []int{
				min(current[0], next[0]),
				min(current[1], next[1]),
				max(current[2], next[2]),
				max(current[3], next[3]),
			}
		} else {
			merged = append(merged, current)
			current = next
		}
	}

	merged = append(merged, current)
	return merged
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
