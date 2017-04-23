#!/usr/bin/python2.7
#import config
from __future__ import print_function
#import sys
#sys.settrace
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
import keras
import errno
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, UpSampling2D, Flatten, Conv1D, MaxPooling1D, Input, ZeroPadding1D, Activation, Dropout
from keras.layers.merge import Concatenate
#from keras.layers import Deconvolution2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.layer_utils import print_summary
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LambdaCallback
import numpy as numpy
import sys
from os import listdir
from os.path import isdir, isfile, islink, join
from random import randint
import random, re, os, time, sys, math, shutil, itertools, argparse
from time import sleep
from math import ceil
import math
from keras.callbacks import EarlyStopping
import shutil
from keras import backend as K
from ansi import *
import cPickle as pickle
import os

def load_abbreviations_re():
	with open(abbrs_file) as f:
		content = f.readlines()
	content = [x.strip() for x in content] 
	#pf(content)
	abbr_re = "(" + '|'.join(content) + ")\."
	#pf(abbr_re)
	#exit(0)
	return re.compile(abbr_re)
#from seya.layers.attention import SpatialTransformer, ST2

def getargs():
	parser = argparse.ArgumentParser(description='Page undeformer')
	parser.add_argument('-f', '--viewfirst', action='store_true', help='View images before training')
	return parser.parse_args()
args = getargs()

def get_linux_terminal():
	env = os.environ
	def ioctl_GWINSZ(fd):
		try:
			import fcntl, termios, struct, os
			cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
		except:
			return
		return cr
	cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
	if not cr:
		try:
			fd = os.open(os.ctermid(), os.O_RDONLY)
			cr = ioctl_GWINSZ(fd)
			os.close(fd)
		except:
			pass
	if not cr:
		cr = (env.get('LINES', 25), env.get('COLUMNS', 80))

		### Use get(key[, default]) instead of a try/catch
		#try:
		#	cr = (env['LINES'], env['COLUMNS'])
		#except:
		#	cr = (25, 80)
	return int(cr[1]), int(cr[0])

input_shape = None
weight_store_txt = "data/weights.h5"
# One level only (we don't go into subdirs)
txt_dir = "data/in"
txt_dir = "data/Gutenberg/txt/"

verbose=0
load_weights=1      # load prior run stored weights
save_weights=1      # load prior run stored weights
test_fraction = .15  # fraction of the data set for the test set
valid_fraction = .1  # fraction of the data set for the validation set
checkpoint_epoch = None
setname_test = 'test'
setname_val = 'val'
setname_train = 'train'

last_epoch_time = time.time()
save_weight_secs = 30
start_time = time.time()
abbrs_file = "abbreviations.txt"
abbrs_re = None # regex for abbreviations

lrate_enh_start = 0.00001
epochs_txt = 15
samp_per_epoch_txt = 200
iters = 50

# 5 10 20 40 80
# 7 14 28 56 112
# Windows like 80, 112
window = 80
punk_max = 4 # Not used currently. This is for outputting a set of punct. offsets,
             # like: (6, 20, ...)
             # while we are currently outputting a one-hot type, like:
			 # (0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0... 1 0 0 0 ...)
actouts=[]  # Activation outputs if viewing layer activations

glob_last_wpunct=None

# Stats
total_sets = total_wpunc = total_wopunc = 0

## Functions
def exit(ec):
	sys.exit(ec)
def pf(*x, **y):
	print(*x, **y)
	sys.stdout.flush()
def pfp(*x, **y):
	y.setdefault('sep', '')
	print(*x, **y)
	sys.stdout.flush()
def pfl(*x, **y):
	y.setdefault('end', '')
	print(*x, **y)
	sys.stdout.flush()
def pfpl(*x, **y):
	y.setdefault('sep', '')
	y.setdefault('end', '')
	print(*x, **y)
	sys.stdout.flush()
def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)
def vprint(verbosity, *args, **kwargs):
	if (verbose >= verbosity):
		pf(*args, **kwargs)

def load_text_dirs():
	global txtsets
	txtall=[]
	vprint(2, "Opening " + txt_dir + "\n")
	i = 0
	for name in listdir(txt_dir):
		vprint(2, "Name:", name)
		i = i+1
		if name.startswith("."):
			vprint(2, bred, "Skipping .hidden file:", name, rst)
		else:
			ifile=join(txt_dir, name)
			vprint(2, "Is file:", ifile)
			if isfile(ifile) or islink(ifile):
				vprint(2, "  appending")
				txtall.append(ifile)
	txtsets = dict()
	cnt = len(txtall)

	s = 0
	e = int(cnt * test_fraction)
	txtsets['test'] = txtall[s : e-1]  # TEST set

	s = e
	e += int(cnt * valid_fraction)
	txtsets['val'] = txtall[s : e-1]   # VALIDATION set

	s = e
	e = cnt
	txtsets['train'] = txtall[s : e-1]

	pf(yel, "Loaded text sets:", len(txtsets['train']), rst)
	pf(gre, "Test text sets:", len(txtsets['test']), rst)
	pf(blu, "Validation text sets:", len(txtsets['val']), rst)

def init():
	# fix random seed for reproducibility
	global termwidth, termheight
	termwidth, termheight = get_linux_terminal()
	seed = 16
	random.seed(seed)
	numpy.random.seed(seed)
	numpy.set_printoptions(threshold=64, linewidth=termwidth-1, edgeitems=3)

	global checkpoint_epoch
	checkpoint_epoch = SaveWeights()

	load_text_dirs()
	global abbrs_re
	abbrs_re = load_abbreviations_re()

	numpy.set_printoptions(edgeitems=100)

def view_weights(model, name=None):
	if name == None:
		raise ValueError("Call with layer")
	layer = model.get_layer(name)
	weights = layer.get_weights()
	pf(weights[1])

def convleaky(inp, filt, dimx=1, ss=(1), name=None, leakalpha=.2, track_list=None, act=None, pad='same', pool=None, poolpad='valid', up=None):
	x = inp
	if not up == None:
		x = UpSampling1D(size=up)(x) # 16
	if not act == None:
		x = Conv1D(filters=filt, kernel_size=dimx, strides=ss, padding=pad, name=name, activation=act)(x)
	else:
		x = Conv1D(filters=filt, kernel_size=dimx, strides=ss, padding=pad, name=name)(x)
		x = LeakyReLU(alpha=leakalpha)(x)
	if track_list: track_list.append({'name':name, 'layer':x})
	if not pool == None:
		x = MaxPooling1D(pool, padding=poolpad)(x)
	return x

def track(name, layer):
	actouts.append({'name':name, 'layer':layer})

class SaveWeights(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		global last_epoch_time
		if time.time()-last_epoch_time > save_weight_secs:
			last_epoch_time = time.time()
			pf("Saving weights, timed (", save_weight_secs, "s).  Time elapsed: ",
					int(time.time()-start_time), "s.  Fit time elapsed: ",
					int(time.time()-fit_start_time), "s.",
					sep=''
				)
			save_weights(model, weight_store_txt)
		return

def show_shape(inputs, x, predict=False):
	# we can predict with the model and print the shape of the array.

	model = Model(inputs=[inputs], outputs=[x])
	pf("MODEL SUMMARY:")
	model.summary()
	pf("/MODEL SUMMARY:")
	if predict:
		dummy_input = numpy.ones((1,window,1), dtype='float32')
		pf("MODEL PREDICT: ",)
		preds = model.predict(dummy_input)
		pf(preds.shape)
		pf("/MODEL PREDICT:")

def model():
	global actouts
	act='LeakyReLU'
	down_track=[]
	up_track=[]
	leakalpha=.2

	x = inputs = Input(shape=(window, 1), name='gen_input', dtype='float32')
	f=64
	#x = Flatten()(x)
	#x = Dense(4096)(x)
	#x = Dense(256)(x)
	#x = Dense(window)(x)
	#x = Reshape((window,))(x)
	#if False:
	if True:
		#x = LeakyReLU(alpha=leakalpha)(x)
		x = convleaky(x, f, 2)    # 80
		x = convleaky(x, f, 3)    # 80
		x = convleaky(x, f, 3)    # 80
		x = MaxPooling1D(2)(x)
		x = convleaky(x, f*2, 2)  # 40
		x = MaxPooling1D(2)(x)
		x = convleaky(x, f*4, 2)  # 20
		x = MaxPooling1D(2)(x)
		x = convleaky(x, f*8, 2)  # 10
		x = MaxPooling1D(2)(x)
		x = convleaky(x, f*16, 2)  # 5
		x = Flatten()(x)
		x = Dense(1024*3)(x)
		#x = Dense(256, activation=act)(x)
		if False: # Use if doing short list of punct. locations
		          #  e.g: (3, 10, 0, 0, 0)
			x = Dense(punk_max)(x)
			x = Reshape((punk_max,))(x)
			x = LeakyReLU(alpha=.2)(x)
		if True:  # use if doing onehot type list of punct. locations
		          #  e.g. (0 0 1 0 0 0 0 0 0 1 0 0 0 0 0...)
			x = Dense(window, activation='sigmoid')(x)
			x = Reshape((window,))(x)
	#show_shape(inputs, x)
	output = x
	actlayers = actmodels = ""
	actlayers = [output] + [ao['layer'] for ao in actouts];
	#down = Model(inputs=[inputs], output=[actlayers])
	#actmodels = Model(inputs=[inputs], output=[actlayers])
	actlosses = [1] + [0 for ao in actouts];
	actmodels = Model(inputs=[inputs], outputs=[output])
	actlosses = [1]
	lrate = lrate_enh_start
	epochs = epochs_txt
	decay = 1/epochs
	adam_opt_gen=Adam(lr=lrate, beta_1=0.6, beta_2=0.999, epsilon=1e-08, decay=decay)
	opt = 'sgd'
	opt = adam_opt_gen
	loss = 'categorical_crossentropy'

	actmodels.compile(
			loss=loss,
			loss_weights=actlosses,
			optimizer='adam',
			metrics=['accuracy'],
		)

	#pf("final prediction: ", sep='', end=''); show_shape(inputs, x)

	#sgd=SGD(lr=0.1, momentum=0.000, decay=0.0, nesterov=False)


	#model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

	pf("Loading weights")
	if load_weights and isfile(weight_store_txt):
		pf("Loading weights")
		actmodels.load_weights(weight_store_txt)
	pf(actmodels.summary())
	return actmodels

def arr1_to_sentence(a):
	#return (a*255).astype(numpy.uint8).tostring()
	return a.astype(numpy.uint8).tostring()

def train(model=None, itercount=0):
	preview = True if args.viewfirst else False
	preview = True
	train = False
	train = True
	#if 1 or (itercount>0 or preview):
	#if 0 and (itercount>0 or preview):
	if itercount>0 or preview:
		generator = generate_texts('test')
		for i in range(0,25):
			x, y = next(generator)
			#pf(bred, "X is ", x, rst)
			#pf(bred, "Y is ", y, rst)
			pred = model.predict(x, batch_size=1, verbose=1)

			pfp("\n  Pred sentence w/punct:\n", glob_last_wpunct)
			s = arr1_to_sentence(x[0])
			pfp("  X sentence:\n", s)
			pfp("  Y gndtrth :", y[0].astype(numpy.uint8))
			# Method: Offsets given as integers:
			#for loc in (((pred[0]*window).astype(numpy.uint8))[::-1]):
				#s = s[:loc] + '\n' + s[loc:]
			# Method: Offsets given as onehot entries where [N]==1 means insert space
			#         at column N
			#pred = y # debug.  REMOVE ME !!!!!!!!!
			#pf("   Y pred    :", pred[0].astype(numpy.uint8))
			pf("   Y pred    :", pred[0].astype(numpy.uint8))
			for loc in range(len(pred[0])-1, -1, -1):
				val = int(pred[0][loc]+.2)  # Add .2 for "close to 1 (true)"
				#pf("pred loc:", val)
				if val: # true == insert space
					s = s[:loc+1] + '/' + s[loc+2:] # use if replacing
					#s = s[:loc+1] + '/' + s[loc+1:] # use if inserting
			pfp("  Y sentence:\n", s)

			#for off in ((pred[0]*window).astype(numpy.int8)[::-1]):
			#for off in ((pred[0]*window).astype(numpy.int8)):
				#for off in (pred[0]):
				#pf(" . at", off)

		#exit(0)
	if not train:
		pf(yel, "Skipping training", rst)
		pf(bred, "Returning early and never running fit_generator", rst)
	if train:
		global fit_start_time, last_epoch_time
		#total_sets = total_wpunc = total_wopunc = 0  # Reset stats
		fit_start_time = time.time()
		last_epoch_time = time.time()

		generator = generate_texts('train')
		generator_val = generate_texts('val')

		pf("Total snippets:", total_sets)
		pf("Total snippets w/ punc:", total_wpunc)
		pf("Total snippets w/o punc:", total_wopunc)
		pf("Calling fit_generator")
		model.fit_generator(
				generator,
				steps_per_epoch=samp_per_epoch_txt,
				epochs=epochs_txt,
				verbose=2,
				validation_data=generator_val,
				validation_steps=30,
				callbacks=[checkpoint_epoch],
			)
		pf("Saving weights")
		save_weights(model, weight_store_txt)

def save_weights(model, fn):
	model.save_weights(fn)
	pf("Saved weights to", fn)

def prep_snippet_in(s):
	global glob_last_wpunct
	snipverbose = True
	snipverbose = False
	#if snipverbose: pfp("String: {{", whi, s, rst, "}}")
	origs = s
	p = re.compile('^\w+\W'); s = p.sub("", s)  # Strip initial word (part)
	p = re.compile('^\W+\s*'); s = p.sub("", s)  # Strip non-word chars
	s = s.lower()
	# White-out(tm) punctuation and symbols
	p = re.compile('[^.!?a-z0-9 ]'); s = p.sub(" ", s)  # .!? should match below
	p = re.compile('\s+'); s = p.sub(" ", s)    # Replace all spaces with single
	s = abbrs_re.sub('\\1', s)                         #   clear . after abbreviations
	#pfp(" After: {{", yel, s, rst, "}}")
	if snipverbose: pfp("  With punct:", bcya, "\n", s[0:135], rst)
	glob_last_wpunct = s
	p = re.compile('(\S)\s+[.?!]\s+'); s = p.sub('\\1.', s)   # should match above
	p = re.compile('[.?!]\s+'); s = p.sub(".", s)   # should match above
	p = re.compile('[.?!]')                         # should match above

	# Find and store punctuation offsets
	starts = [i for i in p.finditer(s)]
	y = numpy.zeros(window)
	i=0
	start_idx = []
	for start in (starts):
		st = start.start()
		if st < window:
			if snipverbose: pf("  Punc found at:", st-i)
			start_idx.append(st-i)
			y[st-1] = 1.0
			#i += 1
	global total_sets, total_wpunc, total_wopunc
	total_sets += 1
	if len(starts):
		total_wpunc += 1
	else:
		total_wopunc += 1
	# White-out that earlier match (the EOL punctuation)
	s = p.sub(" ", s)
	if snipverbose:
		pfp("  Without punct:", yel, "\n", s[0:int(window*1.1)], rst) # crop for display
	s = s[0:window]
	s = s.ljust(window)  # pad with spaces
	

	#pfp(" After: {{", yel, s, rst, "}}")
	s = numpy.fromstring(s, dtype='uint8') # need int8 here to get the chars one at a time
	s = s.astype(numpy.float32)
	#s = s/255
	#pf("s scaled:", s)
	s = s.reshape((1,) + s.shape + (1,))
	start_idx = start_idx[:punk_max]
	start_idx.extend([0] * (punk_max - len(start_idx)))
	start_idx = numpy.asarray(start_idx, dtype='float32')
	#pf("Cleaned S:", yel, cleans[:140], rst)
	if snipverbose: pf("Indexes       :", start_idx)
	start_idx = start_idx / window
	#pf("Indexes scaled:", start_idx)
	start_idx = start_idx.reshape((1,punk_max))
	#pf("Orig S:", bgre, origs[:150], rst)
	#pf("Punk locs::", bcya, start_idx, rst)
	#pf("shhh", start_idx.shape)
	if snipverbose: pf("")
	#return s, start_idx
	y = y.reshape((1,window))
	#if snipverbose: pf("String        :", s)
	if snipverbose: pf("Indexes       :", y)
	#if snipverbose: pf("Indexes shape :", y.shape)
	return s, y
def get_snippet(fn):
	try:
		fstat = os.stat(fn)
		flen = fstat.st_size
		if flen < window:
			raise ValueError("File " + fn + " is too short: " + str(flen) + "<" + str(window))
		start = randint(0, flen-(window*2))  # we might not get a last phrase(s) here
		#pf("File:", fn)
		f = open(fn, "rb")
		f.seek(start)
		# Read twice the amount so, hopefully, we can find word boundaries
		# and strip punctuation, and have window (128) chars left
		data = f.read(window*2)
		f.close()
		string,punclocs = prep_snippet_in(data)
		#pf("Punct locs  :", punclocs)
		#pf("np len int8:", string.shape)
		#pf("np len float32:", string.shape)
		#pf("Shape x:", string.shape)
		#pf(string)
		return string,punclocs
	except:
		raise

def generate_texts(setname): # setname='train','val','test'
	global lastfname
	while True:
		iset = txtsets[setname]
		iidx = randint(0, len(iset)-1)
		lastfname = iset[iidx]
		x, y = get_snippet(iset[iidx])
		#pf("X:", x)
		#pf("Y:", y)
		#pf("Ys:", y.shape)
		yield x, y

init()
model = model()

uncolor()
for i in xrange(iters):
	train(model=model, itercount=i)

#if not args.hyperenh:
#unbents, flats = zip(itertools.islice(gen_img_enh_pair('test'), 100))
#pf("Evalutation of best performing model:")
#pf(best_model.evaluate(unbents, flats))
#pf("Best performing model chosen hyper-parameters:")
#pf(best_run)

pf("Enter to close")
inp=raw_input('')
exit(0)
	
# vim:ts=4 ai
