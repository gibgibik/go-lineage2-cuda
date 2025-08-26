package cmd

import (
	"context"
	"fmt"
	"github.com/gibgibik/go-lineage2-cuda/cmd/web"
	"github.com/gibgibik/go-lineage2-cuda/internal/core"
	"github.com/spf13/cobra"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"os"
	"os/signal"
	"syscall"
)

func Execute() error {
	var err error
	pe := zap.NewProductionEncoderConfig()
	pe.EncodeTime = zapcore.ISO8601TimeEncoder
	consoleEncoder := zapcore.NewConsoleEncoder(pe)

	wsEncoded := zap.NewDevelopmentEncoderConfig()
	wsEncoded.EncodeTime = zapcore.RFC3339TimeEncoder

	webEncoder := zap.NewDevelopmentEncoderConfig()
	webEncoder.EncodeLevel = func(l zapcore.Level, enc zapcore.PrimitiveArrayEncoder) {
		l.Set("")
	}
	webEncoder.EncodeTime = zapcore.TimeEncoderOfLayout("15:04:05")

	f, err := os.OpenFile(fmt.Sprintf("var/log/app.log"), os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0600)
	if err != nil {
		panic(err)
	}
	w := zapcore.AddSync(f)
	cZ := zapcore.NewTee(
		zapcore.NewCore(
			zapcore.NewJSONEncoder(pe),
			w,
			zap.InfoLevel,
		),
		zapcore.NewCore(consoleEncoder, zapcore.AddSync(os.Stdout), zapcore.DebugLevel),
	)
	logger := zap.New(cZ)
	cnf, err := core.InitConfig()
	if err != nil {
		return err
	}
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT, syscall.SIGKILL)
	defer cancel()

	rootCmd := &cobra.Command{
		PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
			return nil
		},
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmd.Usage()
		},
	}
	rootCmd.AddCommand(web.CreateWebServerCommand(logger.Sugar()))
	go func() {
		defer cancel()
		err = rootCmd.ExecuteContext(context.WithValue(ctx, "cnf", cnf))
	}()
	<-ctx.Done()
	return err
}
