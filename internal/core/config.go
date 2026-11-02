package core

import (
	"github.com/spf13/viper"
)

type Web struct {
	Port string
}
type ScrydeCaptcha struct {
	MaskFile string `mapstructure:"mask_file"`
	InputCoordinates  []int `mapstructure:"input_coordinates"`
	SubmitCoordinates []int `mapstructure:"submit_coordinates"`
}
type Config struct {
	WebServer     Web                      `mapstructure:"web"`
	BaseUrl       string                   `mapstructure:"base_url"`
	ScrydeCaptcha map[string]ScrydeCaptcha `mapstructure:"scryde_captcha"`
}

func InitConfig() (*Config, error) {
	viper.SetConfigFile("configs/main.yaml")
	if err := viper.ReadInConfig(); err != nil {
		return nil, err
	}
	v := viper.New()
	if v.ReadInConfig() == nil {
		err := viper.MergeConfigMap(v.AllSettings())
		if err != nil {
			return nil, err
		}
	}
	config := &Config{}
	if err := viper.Unmarshal(config); err != nil {
		return nil, err
	}

	return config, nil
}
