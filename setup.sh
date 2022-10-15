mkdir images
sudo docker-compose up -d

echo "You may now place images to encode in the images/ folder."
echo "Then run encode_and_insert.py to store the embeddings in MilvusDB."
echo "Finally, search_face.py can be used on a new image to search for hits from DB."