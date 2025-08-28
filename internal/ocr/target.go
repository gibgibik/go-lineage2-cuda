package ocr

import (
	"github.com/doraemonkeys/paddleocr"
	"go.uber.org/zap"
	"sync"
)

type Target struct {
	sync.Mutex
	Cl *paddleocr.Ppocr
}

func NewTarget(logger *zap.SugaredLogger) *Target {
	p, err := paddleocr.NewPpocr("/opt/paddleocr/bin/PaddleOCR-json",
		paddleocr.OcrArgs{}, "-models_path", "/opt/paddleocr/models")
	if err != nil {
		panic(err)
	}
	return &Target{
		Cl: p,
	}
}
