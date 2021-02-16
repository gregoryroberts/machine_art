#
# Note: this code is a mix of my own and some from the linked pytorch tutorial (whether or in inspiration or copeid from there).
# I have attempted to note main places in the code where I leaned most heavily on the tutorial, but it is possible I have missed
# some parts.
# The data provided for the images and the style are my own, but the VGG-19 network used is loaded in pretrained from pytorch.
# To my knowledge, it is trained on ImageNet's database.  Link to pytorch tutorial:
# https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
#

#
# System
#
import sys

#
# Math
#
import numpy as np

#
# Neural
#
import torch
from torchvision import transforms
import torch.nn.functional as F

#
# Viz and Image Processing
#
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import kornia
import imageio


#
# Setup output structures and paths for writing final .gif
#
sunset_gif_art_images = []
sunset_gif_content_images = []
sunset_gif_style_images = []

sunset_gif_art_path = '../output/sunset/art.gif'
sunset_gif_content_path = '../output/sunset/content.gif'
sunset_gif_style_path = '../output/sunset/style.gif'

num_sunset_images = 9

def batch_to_image( pytorch_batch, viz=False ):
	get_np_array = np.swapaxes( np.swapaxes( np.squeeze( pytorch_batch.detach().numpy() ), 0, 2 ), 0, 1 )
	
	if viz:
		plt.imshow( get_np_array )
		plt.show()

	return Image.fromarray( np.uint8( 255 * get_np_array ) )

#
# Setup pytorch VGG pretrained model on ImageNet
#
model = torch.hub.load( 'pytorch/vision:v0.6.0', 'vgg19', pretrained=True )
model.eval()

for sunset_img_idx in range( 0, num_sunset_images ):

	sunset_image = Image.open( '../data/sunset/sunset_' + str( sunset_img_idx ) + '.jpg' )
	sunset_style = Image.open( '../data/sunset/style_' + str( sunset_img_idx ) + '.jpg' )

	renormalize = transforms.Normalize( mean=[ -0.485 / .229, -0.456 / .224, -0.406 / .225 ], std=[ 1. / 0.229, 1. / 0.224, 1. / 0.225 ] )

	def gen_preprocess( img ):
		min_dim = np.minimum( img.size[ 0 ], img.size[ 1 ] )
		print( min_dim )

		preprocessor = transforms.Compose([
			transforms.CenterCrop( min_dim ),
			transforms.Resize( 256 ),
			transforms.CenterCrop( 224 ),
			transforms.ToTensor(),
			transforms.Normalize( mean=[ 0.485, 0.456, 0.406 ], std=[ 0.229, 0.224, 0.225 ] ),
		])

		return preprocessor

	def gen_preprocess_no_normalize( img ):
		min_dim = np.minimum( img.size[ 0 ], img.size[ 1 ] )
		print( min_dim )

		preprocessor = transforms.Compose([
			transforms.CenterCrop( min_dim ),
			transforms.Resize( 256 ),
			transforms.CenterCrop( 224 ),
			transforms.ToTensor(),
		])

		return preprocessor

	sunset_image_prepocess = gen_preprocess( sunset_image )
	sunset_style_preprocess = gen_preprocess( sunset_style )

	sunset_image_prepocess_no_normalize = gen_preprocess_no_normalize( sunset_image )
	sunset_style_preprocess_no_normalize = gen_preprocess_no_normalize( sunset_style )


	style_tensor = sunset_style_preprocess( sunset_style )
	style_batch = style_tensor.unsqueeze( 0 )

	style_tensor_no_normalize = sunset_style_preprocess_no_normalize( sunset_style )
	style_batch_no_normalize = style_tensor_no_normalize.unsqueeze( 0 )

	content_tensor = sunset_image_prepocess( sunset_image )
	content_batch = content_tensor.unsqueeze( 0 )

	content_tensor_no_normalize = sunset_image_prepocess_no_normalize( sunset_image )
	content_batch_no_normalize = content_tensor_no_normalize.unsqueeze( 0 )

	composition_layers = [ 0, 2 ]
	style_layers = [ 2, 7, 10, 12 ]

	max_network_idx = np.maximum( np.max( composition_layers ), np.max( style_layers ) )


	#
	# From pytorch tutorial! Faster than my implementation and also works with autograd!
	# The placeholder classes below are also in the style of the tutorial, which gives a nice
	# way to track loss functions with respect to different feature maps.
	# Source code credit: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
	#
	def gram_matrix( input ):
		a, b, c, d = input.size()  # a=batch size(=1)
		# b=number of feature maps
		# (c,d)=dimensions of a f. map (N=c*d)

		features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

		G = torch.mm( features, features.t() )  # compute the gram product

		# we 'normalize' the values of the gram matrix
		# by dividing by the number of element in each feature maps.
		return G.div( a * b * c * d )


	class CompositionPlaceholder( torch.nn.Module ):
		def __init__( self, target_composition ):
			super( CompositionPlaceholder, self ).__init__()

			self.target_composition = target_composition.detach()

		def forward( self, x ):
			self.loss = F.mse_loss( x, self.target_composition )

			return x

	class StylePlaceholder( torch.nn.Module ):
		def __init__( self, target_features ):
			super( StylePlaceholder, self ).__init__()

			self.target_style_correlation = gram_matrix( target_features.detach() )

		def forward( self, x ):
			correlation = gram_matrix( x )
			self.loss = F.mse_loss( correlation, self.target_style_correlation )

			return x

	replace_max_pooling_with_average_pooling = True
	model_layer_idx = 0
	new_model = []

	progress_content = content_batch.detach().clone()
	progress_style = style_batch.detach().clone()

	content_placeholder_layers = []
	style_placeholder_layers = []

	for model_layer in model.features:

		choose_model_layer = model_layer
		if replace_max_pooling_with_average_pooling:
			if isinstance( model_layer, torch.nn.MaxPool2d ):
				new_model_layer = torch.nn.AvgPool2d(
					kernel_size=model_layer.kernel_size,
					stride=model_layer.stride,
					padding=model_layer.padding,
					ceil_mode=model_layer.ceil_mode )

				choose_model_layer = new_model_layer
			if isinstance( model_layer, torch.nn.ReLU ):
				choose_model_layer = torch.nn.ReLU( inplace=False )

		new_model.append( choose_model_layer )

		progress_content = choose_model_layer.forward( progress_content )
		progress_style = choose_model_layer.forward( progress_style )

		if model_layer_idx in composition_layers:
			content_placeholder_layers.append( CompositionPlaceholder( progress_content ) )
			new_model.append( content_placeholder_layers[ -1 ] )
		if model_layer_idx in style_layers:
			style_placeholder_layers.append( StylePlaceholder( progress_style ) )
			new_model.append( style_placeholder_layers[ -1 ] )

		if model_layer_idx == max_network_idx:
			break

		model_layer_idx += 1


	sub_model = torch.nn.Sequential( *new_model )

	print( sub_model )

	num_epochs = 300

	#
	# Turn off gradient for model parameters, and just optimize the random image
	#
	for params in sub_model.parameters():
		params.requires_grad = False

	net_norm = transforms.Normalize( mean=[ 0.485, 0.456, 0.406 ], std=[ 0.229, 0.224, 0.225 ] )

	style_weight = 1e6
	composition_weight = 20

	use_random_seed = True

	if use_random_seed:
		torch.manual_seed( 5234234 )

		# random_image = torch.rand( content_batch.shape, requires_grad=True )
		random_image = torch.ones( content_batch.shape, requires_grad=True )
		random_image = random_image.detach() * 0.5
		# gauss = kornia.filters.GaussianBlur2d( ( 5, 5 ), ( 5, 5 ) )
		# random_image = gauss( random_image.detach().float() )

	else:
		random_image = renormalize( content_batch )# torch.rand( content_batch.shape, requires_grad=True )


	init_random_img = random_image.clone()

	optimizer = torch.optim.LBFGS( [ random_image.requires_grad_() ] )

	epoch_idx = [ 0 ]
	while epoch_idx[ 0 ] < num_epochs:

		#
		# Inspired by pytorch tutorial (especially LBFGS optimizer)! This tutorial also helps pick content vs. style weights because
		# they are on very different orders of magnitude
		# Source code credit: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
		#
		def closure():

			random_image.data.clamp( 0, 1 )

			optimizer.zero_grad()

			feature_output = sub_model( net_norm.forward( random_image ) )

			composition_loss = 0
			for content_layer in content_placeholder_layers:
				composition_loss += content_layer.loss

			style_loss = 0
			for style_layer in style_placeholder_layers:
				style_loss += style_layer.loss

			combined_loss = style_weight * style_loss + composition_weight * composition_loss

			combined_loss.backward()

			if ( epoch_idx[ 0 ] % 10 ) == 0:
				print( 'On epoch ' + str( epoch_idx ) + ', the current composition loss value is ' + str( composition_weight * composition_loss.detach().numpy() ) )
				print( 'On epoch ' + str( epoch_idx ) + ', the current correlation loss value is ' + str( style_weight * style_loss.detach().numpy() ) )
				print( 'On epoch ' + str( epoch_idx ) + ', the combined loss value is ' + str( combined_loss.detach().numpy() ) )
				print()

			epoch_idx[ 0 ] += 1

			return combined_loss


		optimizer.step( closure )


	random_image.data.clamp_(0, 1)

	display_image = random_image
	display_seed = init_random_img

	sunset_gif_art_images.append( batch_to_image( random_image, viz=False ) )
	sunset_gif_content_images.append( batch_to_image( content_batch_no_normalize, viz=False ) )
	sunset_gif_style_images.append( batch_to_image( style_batch_no_normalize, viz=False ) )



seconds_per_image = 33
imageio.mimsave( sunset_gif_art_path, sunset_gif_art_images, duration=seconds_per_image )
imageio.mimsave( sunset_gif_content_path, sunset_gif_content_images, duration=seconds_per_image )
imageio.mimsave( sunset_gif_style_path, sunset_gif_style_images, duration=seconds_per_image )


