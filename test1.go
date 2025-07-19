package main

import (
	"fmt"
	"gocv.io/x/gocv"
	"image"
	"image/color"
)

func main() {
	model := "dbnet.onnx" // завантаж модель сюди
	inputPath := "1.png"
	outputPath := "output.png"

	// Завантаження моделі
	net := gocv.ReadNet(model, "input")
	if net.Empty() {
		panic("❌ Не вдалося завантажити модель")
	}
	defer net.Close()

	// Вмикаємо CUDA
	net.SetPreferableBackend(gocv.NetBackendCUDA)
	net.SetPreferableTarget(gocv.NetTargetCUDA)

	// Завантажуємо зображення
	img := gocv.IMRead(inputPath, gocv.IMReadColor)
	if img.Empty() {
		panic("❌ Не вдалося завантажити зображення")
	}
	defer img.Close()

	// Розмір вхідного зображення для DBNet
	inputSize := image.Pt(736, 736)
	blob := gocv.BlobFromImage(img, 1.0/255.0, inputSize, gocv.NewScalar(0, 0, 0, 0), true, false)
	defer blob.Close()

	net.SetInput(blob, "input")

	// Проганяємо forward
	result := net.Forward("logits")
	defer result.Close()

	dims := result.Size()
	fmt.Println("🧠 Output shape:", dims)
	if len(dims) != 4 || dims[1] != 1 {
		panic(fmt.Sprintf("❌ Невірна форма виходу: %v", dims))
	}

	// Отримуємо 2D карту ймовірностей
	channels := gocv.Split(result)
	if len(channels) < 1 {
		panic("❌ Неможливо розділити канал результату")
	}
	probMap := channels[0]
	defer probMap.Close()

	// Ресайз до розміру оригінального зображення
	gocv.Resize(probMap, &probMap, image.Pt(img.Cols(), img.Rows()), 0, 0, gocv.InterpolationLinear)

	// Бінаризація (поріг 0.3)
	binary := gocv.NewMat()
	defer binary.Close()
	gocv.Threshold(probMap, &binary, 0.3, 255, gocv.ThresholdBinary)

	// Знаходимо контури
	contours := gocv.FindContours(binary, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer contours.Close()

	// Малюємо рамки
	for i := 0; i < contours.Size(); i++ {
		c := contours.At(i)
		rect := gocv.BoundingRect(c)
		gocv.Rectangle(&img, rect, color.RGBA{0, 255, 0, 0}, 2)
	}

	// Зберігаємо результат
	ok := gocv.IMWrite(outputPath, img)
	if !ok {
		panic("❌ Не вдалося зберегти вихідний файл")
	}

	fmt.Println("✅ Готово! Збережено:", outputPath)
}
