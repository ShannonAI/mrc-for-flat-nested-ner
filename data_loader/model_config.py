#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Description:
# config files used for model init 


import json 
import copy 



class Config(object):

    @classmethod 
    def from_dict(cls, json_object):
        config_instance = Config()
        for key, value in json_object.items():
            try:
                tmp_value = Config.from_json_dict(value)
                config_instance.__dict__[key] = tmp_value 
            except:
                config_instance.__dict__[key] = value 
        return config_instance 

    @classmethod 
    def from_json_file(cls, json_file):
        with open(json_file, "r") as f:
            text = f.read()
        return Config.from_dict(json.loads(text))


    @classmethod 
    def from_json_dict(cls, json_str):
        return Config.from_dict(json_str)


    @classmethod 
    def from_json_str(cls, json_str):
        return Config.from_dict(json.loads(json_str))


    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output = {k: v.to_dict() if isinstance(v, Config) else v for k, v in output.items()}
        return output 


    def print_config(self):
        model_config = self.to_dict()
        json_config = json.dumps(model_config, indent=2)
        print(json_config)
        return json_config 


    def to_json_string(self):
        return json.dumps(self.to_dict(), indent = 2) + "\n"


    def update_args(self, args_namespace):
        args_dict = args_namespace.__dict__
        print("Please notice that merge the args_dict and json_config ... ...")
        for args_key, args_value in args_dict.items():
            if args_key not in self.__dict__.keys():
                self.__dict__[args_key] = args_value 
            else:
                print("update the config from args input ... ...")
                self.__dict__[args_key] = args_values 



if __name__ == "__main__":
    str_instance = "{'name': 'lixiaoya'}"
    json_config = json.loads(str_instance)
    print("check the content of json_files")
    print(json_config.keys())
    
