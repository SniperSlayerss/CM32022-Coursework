zip: 
	zip Group_C_code.zip -r ./ -s 45m --exclude ".git/*" --exclude ".direnv/*" --exclude ".venv/*" --exclude "./**/__pycache__/*" --exclude "./.ruff_cache/*"

clean:
	rm *.z*
