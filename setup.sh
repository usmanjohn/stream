mkdir - p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = True\n\
\n\
" > ~/.streamlit/config.toml
