import csv
import requests
import os

from PIL import Image
from io import BytesIO


CSV_FILE = os.path.abspath('./data/MovieGenre.csv')
IMAGES_PATH = os.path.join(os.path.abspath('./data'), 'IMDB_IMAGES')

if (os.path.exists(IMAGES_PATH) == False):
	os.mkdir(IMAGES_PATH)
	
with open(CSV_FILE, 'r', encoding='utf-8') as file:
	reader = csv.reader(file)
	
	# skip head 
	next(reader)

	for row in reader:
		imdb_link = row[1]
		image_url = row[5]

		# Vérification de l'URL
		if not image_url:
			print(f"L'URL de l'image est vide pour la ligne {reader.line_num}.")
			continue

		# Ajout du préfixe 'http://' si aucun schéma n'est présent dans l'URL
		if not image_url.startswith('http'):
			image_url = 'http://' + image_url

		# Téléchargement de l'image
		try:
			response = requests.get(image_url)
		except requests.exceptions.RequestException as e:
			print(f"Erreur lors du téléchargement de l'image à l'URL {image_url}: {e}.")
			continue

		# Vérification du contenu de la réponse
		if 'image' not in response.headers.get('content-type', '').lower():
			print(f"Impossible de télécharger l'image à l'URL {image_url}.")
			continue

		# Ouverture de l'image à l'aide de PIL
		try:
			image = Image.open(BytesIO(response.content))
		except (OSError, IOError):
			print(f"Impossible d'ouvrir l'image à l'URL {image_url}.")
			continue
		
		resized_image = image.resize((255, 255))

		# Enregistrement de l'image redimensionnée
		image_filename = f"{row[0]}.jpg"  # Utilise imdbId comme nom de fichier
		resized_image.save(os.path.join(IMAGES_PATH, image_filename))

		print(f"Image {image_filename} téléchargée avec succès.")