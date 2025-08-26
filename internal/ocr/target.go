package ocr

import (
	"github.com/otiai10/gosseract/v2"
	"go.uber.org/zap"
	"sync"
)

type Target struct {
	sync.Mutex
	Cl *gosseract.Client
}

func NewTarget(logger *zap.SugaredLogger) *Target {
	return &Target{
		Cl: gosseract.NewClient(),
	}
}
