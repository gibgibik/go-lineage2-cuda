package core

import "go.uber.org/zap"

type FwdToZapWriter struct {
	Logger *zap.SugaredLogger
}

func (fw *FwdToZapWriter) Write(p []byte) (n int, err error) {
	fw.Logger.Errorw(string(p))
	return len(p), nil
}
