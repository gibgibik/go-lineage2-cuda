package ocr

import (
	"fmt"
	"gocv.io/x/gocv"
)

func InitNet() gocv.Net {
	net := gocv.ReadNet("var/frozen_east_text_detection.pb", "")
	if net.Empty() {
		fmt.Println("Error reading network model")
		panic("Error reading network model")
	}
	err := net.SetPreferableBackend(gocv.NetBackendCUDA)
	if err != nil {
		fmt.Println("Error setting preferable target")
		panic(err)
	}
	err = net.SetPreferableTarget(gocv.NetTargetCUDA)
	if err != nil {
		fmt.Println("Error setting preferable target")
		panic(err)
	}

	return net
}
