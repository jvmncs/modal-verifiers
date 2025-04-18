{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      cuda = pkgs.cudaPackages_12_4;
    in {

      devShells.${system}.default = pkgs.mkShell {
        nativeBuildInputs = with cuda; [
          cudatoolkit
          cuda_nvrtc
          cuda_cupti
          cudnn
        ];

        shellHook = with cuda; ''
            export CUDA_PATH=${cudatoolkit}
            export CUDA_HOME=${cudatoolkit}
            export LD_LIBRARY_PATH=${cudatoolkit}/lib:${cudnn}/lib:${cuda_nvrtc}/lib:${cuda_cupti}/lib:$LD_LIBRARY_PATH
            export XLA_FLAGS="--xla_gpu_cuda_data_dir=${cudatoolkit}"

            export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=/run/opengl-driver/lib64:$LD_LIBRARY_PATH

            echo "CUDA ${cudatoolkit.version} environment activated"
        '';
      };
    };
}
