{
  description = "cv coursework flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/25.11";
  };

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      stdenv = pkgs.stdenv;
      lib = pkgs.lib;
    in {
      devShells.${system}.default = pkgs.mkShell{
        name = "cvcw";

        NIX_LD_LIBRARY_PATH = lib.makeLibraryPath [
          stdenv.cc.cc
          pkgs.zlib
          pkgs.libxcb
          pkgs.libGL
          pkgs.glib
        ];

        NIX_LD = lib.fileContents "${stdenv.cc}/nix-support/dynamic-linker";

        packages = with pkgs; [
          python3
          uv
        ];

        shellHook = ''
          export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH
        '';
      };
    };
}

