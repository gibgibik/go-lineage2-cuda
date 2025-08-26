package web

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/gibgibik/go-lineage2-cuda/internal/core"
	"github.com/gibgibik/go-lineage2-cuda/internal/ocr"
	"github.com/spf13/cobra"
	"go.uber.org/zap"
	"io"
	"log"
	"net"
	"net/http"
	"time"
)

var (
	targetCl *ocr.Target
)

func CreateWebServerCommand(logger *zap.SugaredLogger) *cobra.Command {
	var webServer = &cobra.Command{
		Use: "web-server",
		RunE: func(cmd *cobra.Command, args []string) error {
			cnf := cmd.Context().Value("cnf").(*core.Config)
			handle := httpServerStart(cmd.Context(), cnf, logger)
			for {
				select {
				case <-cmd.Context().Done():
					err := handle.Shutdown(cmd.Context())
					if targetCl != nil {
						_ = targetCl.Cl.Close()
					}
					logger.Info(fmt.Sprintf("web-server stop result: %v", err))
					return err
				default:
					time.Sleep(time.Microsecond * 100000)
				}
			}
		},
	}
	return webServer
}

func httpServerStart(ctx context.Context, cnf *core.Config, logger *zap.SugaredLogger) *http.Server {
	logger.Debug("starting webserver on port :", cnf.WebServer.Port)
	mux := http.NewServeMux() // Create
	mux.HandleFunc("/findBounds", findBoundsHandler(logger))
	mux.HandleFunc("/findTargetName", findTargetNameHandler(logger))
	handle := &http.Server{
		Addr:         ":" + cnf.WebServer.Port,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
		ErrorLog:     log.New(&core.FwdToZapWriter{logger}, "", 0),
		BaseContext: func(listener net.Listener) context.Context {
			return context.WithValue(ctx, "logger", logger)
		},
		Handler: mux,
	}
	//mux.HandleFunc("/ws", wsHandler)
	//mux.HandleFunc("/api/profile", getProfilesListHandler(logger))
	//mux.HandleFunc("/api/profile/", templateHandler)
	//mux.HandleFunc("/api/start/", startHandler(ctx, cnf))
	//mux.HandleFunc("/api/pause", pauseHandler())
	//mux.HandleFunc("/api/stop", stopHandler())
	//mux.HandleFunc("/api/init", initHandler())
	//mux.HandleFunc("/api/preset", getPresetsListHandler(logger))
	//mux.HandleFunc("/api/preset/", savePresetHandler(logger))
	//mux.HandleFunc("/api/stats", statHandler(logger))
	//mux.HandleFunc("/api/npc", npcHandler(logger))
	//mux.Handle("/", http.FileServer(http.Dir("./web/dist")))
	go func() {
		if err := handle.ListenAndServe(); err != nil {
			if !errors.Is(err, http.ErrServerClosed) {
				logger.Error("http server fatal error: " + err.Error())
			}
		}
	}()

	return handle
}

func findBoundsHandler(logger *zap.SugaredLogger) func(writer http.ResponseWriter, request *http.Request) {
	return func(writer http.ResponseWriter, request *http.Request) {
		var imgB, err = io.ReadAll(request.Body)
		if err != nil {
			errM := "fail to read body"
			logger.Error(errM)
			createRequestError(writer, errM, http.StatusBadRequest)
			return
		}
		defer request.Body.Close()
		fmt.Println(imgB)
	}
}

func findTargetNameHandler(logger *zap.SugaredLogger) func(writer http.ResponseWriter, request *http.Request) {
	targetCl = ocr.NewTarget(logger)
	return func(writer http.ResponseWriter, request *http.Request) {
		var imgB, err = io.ReadAll(request.Body)
		if err != nil {
			errM := "fail to read body"
			logger.Error(errM)
			createRequestError(writer, errM, http.StatusBadRequest)
			return
		}
		defer request.Body.Close()
		targetCl.Lock()
		defer targetCl.Unlock()
		err = targetCl.Cl.SetImageFromBytes(imgB)
		if err != nil {
			logger.Error(err.Error())
			createRequestError(writer, err.Error(), http.StatusBadRequest)
			return
		}
		text, err := targetCl.Cl.Text()
		if err != nil {
			logger.Error(err.Error())
			createRequestError(writer, err.Error(), http.StatusBadRequest)
			return
		}
		res := struct {
			Name string `json:"name"`
		}{
			Name: text,
		}
		j, _ := json.Marshal(res)
		writer.Write(j)
		return
	}
}

func createRequestError(w http.ResponseWriter, err string, code int) {
	w.WriteHeader(code)
	_, _ = w.Write([]byte(err))
}
