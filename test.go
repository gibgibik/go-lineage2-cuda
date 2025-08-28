package main

import (
	"fmt"
	"github.com/doraemonkeys/paddleocr"
	"time"
)

func main() {
	p, err := paddleocr.NewPpocr("/opt/paddleocr/bin/PaddleOCR-json",
		paddleocr.OcrArgs{}, "-models_path", "/opt/paddleocr/models")
	if err != nil {
		panic(err)
	}
	defer p.Close()
	start := time.Now()
	result, err := p.OcrFileAndParse(`/var/www/go-lineage2-server/123.png`)
	elapsed := time.Since(start)
	fmt.Printf("Execution name took %s\n", elapsed)
	if err != nil {
		panic(err)
	}
	if result.Code != paddleocr.CodeSuccess {
		fmt.Println("orc failed:", result.Msg)
		return
	}
	fmt.Println(result.Data)
}
