docker build -t longtou/cosyvoice:aihub .
docker tag longtou/cosyvoice:aihub us-central1-docker.pkg.dev/prod-ai-project/tts/cosyvoice:aihub

#docker run -it --runtime=nvidia longtou/cosyvoice:aihub /bin/bash
#gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://us-central1-docker.pkg.dev
#docker push us-central1-docker.pkg.dev/prod-ai-project/tts/cosyvoice:aihub
