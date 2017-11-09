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
from keras.layers import Dense, Reshape, UpSampling2D, Flatten, Conv1D, MaxPooling1D, Input, ZeroPadding1D, Activation, Dropout, Embedding, Permute, LSTM
from keras.layers.merge import Concatenate
#from keras.layers import Deconvolution2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.layer_utils import print_summary
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.utils.np_utils import to_categorical
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
from utils import * # load_abbreviations_re():
import cPickle as pickle
import os

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
abbrs_re = None # regex for abbreviations

lrate_enh_start = 0.001
epochs_txt = 20
samp_per_epoch_txt = 20000
iters = 50

# 5 10 20 40 80
# 7 14 28 56 112
# Windows like 80, 112
window = 17
punk_max = 4 # Not used currently. This is for outputting a set of punct. offsets,
             # like: (6, 20, ...)
             # while we are currently outputting an array with 1's at the offsets.
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
	if not track_list == None:
		#if name == None: raise ValueError('track_list given, but no name')
		track_list.append({'name':name, 'layer':x})
	if not pool == None:
		x = MaxPooling1D(pool, padding=poolpad)(x)
	return x

def track_add(track_list, name, layer):
	track_list.append({'name':name, 'layer':layer})

class SaveWeights(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		global last_epoch_time
		sleep(.5)
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
	act='sigmoid'
	trackers=[]
	leakalpha=.2

	x = inputs = Input(shape=(window, 1), name='gen_input', dtype='float32')
	#track_add(trackers, 'gen_input', x)
	#x = Flatten()(x)
	#x = Dense(4096)(x)
	#x = Dense(256)(x)
	#x = Dense(window)(x)
	#x = Reshape((window,))(x)
	#if False:
	f=10
	if True:
		#x = LeakyReLU(alpha=leakalpha)(x)
		charset_in=256   # latin-1?
		charset_out=62  # Enough?
		x = Reshape((window,))(x)
		x = Embedding(charset_in, charset_out, input_length=window)(x)
		x = LSTM(window*charset_out, dropout=0.2, recurrent_dropout=0.2)(x)
		x = Dense(window, activation='sigmoid')(x)
		#x = LeakyReLU(alpha=.2)(x)
		#x = Reshape((window,))(x)
#		x = convleaky(x, 50, 1, track_list=trackers, name='c1_1')    # 80
#		x = convleaky(x, 50, 1, track_list=trackers, name='c1_2', act=act)
#		x = convleaky(x, f, 5, track_list=trackers, name='c1_3', act=act)
#		x = MaxPooling1D(2)(x)
#		x = convleaky(x, f*2, 2, act=act)  # 40
#		x = MaxPooling1D(2)(x)
#		x = convleaky(x, f*4, 2, act=act)  # 20
#		x = MaxPooling1D(2)(x)
#		x = convleaky(x, f*8, 2, act=act)  # 10
#		x = MaxPooling1D(2)(x)
#		x = convleaky(x, f*16, 2, act=act)  # 5
#		x = MaxPooling1D(2)(x)
#		x = convleaky(x, f*32, 2, act=act)  # 5
#		x = Flatten()(x)
#		#x = Dense(1024*3, activation=act)(x)
#		x = Dense(window, activation='sigmoid')(x)
#		x = Reshape((window,))(x)
	#show_shape(inputs, x)
	output = x
	actmodels = ""
	actlayers = [output] + [track['layer'] for track in trackers];
	pf("Tracking layers as outputs:")
	pf("  output")
	[ pf(" ", track['name']) for track in trackers ]
	pf("Inputs:", inputs)
	actlosses = [1] + [0 for track in trackers];
	actmodels = Model(inputs=[inputs], outputs=actlayers)
	lrate = lrate_enh_start
	epochs = epochs_txt
	decay = 1/epochs
	adam_opt_gen=Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
	opt = 'sgd'
	opt = adam_opt_gen
	loss = 'binary_crossentropy'
	loss = 'categorical_crossentropy'

	pf("Actmodels Model:", actmodels)
	actmodels.compile(
			loss=loss,
			loss_weights=actlosses,
			optimizer=opt,
			#metrics=['accuracy'],
			metrics=['categorical_accuracy'],
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
	do_train = False
	do_train = True
	#if 1 or (itercount>0 or preview):
	#if 0 and (itercount>0 or preview):
	if itercount>0 or preview:
		generator = generate_texts_rnd('test')
		for i in range(0,5):
			x, y = next(generator)
			#pf(bred, "X is ", x, rst)
			#pf(bred, "Y is ", y, rst)
			pred = model.predict(x, batch_size=1, verbose=0)
			#pf(pred)
			# pred[2] is the 3rd output: c1_2
			# pred[2][0] is the first sample (of the batch)'s output
			# pred[2][0][i] is the first sample output's letter i
			# pred[2][0][i][0] is letter's conv filter (there are f = 100)
			#pf("Pred len:", len(pred))
			#pf("Pred[0] shape (output layer):", pred[0].shape)
			#pf("Pred[1] shape:", pred[1].shape)
			#pf("Pred[2] shape:", pred[2].shape)

			pfp("\n  Pred sentence w/punct:\n", glob_last_wpunct)
			s = arr1_to_sentence(x[0])
			pfp("  Pred sentence input (x):\n", s)

			#lyr_c1_1_out = pred[2][0] # Output#2, for the first of batch (0)
			#for f in range(len(lyr_c1_1_out[0])):  # f = 0..filter count
			#	lyr_c1_1_out = pred[2][0] # Output#2, for the first of batch (0)
			#	ltr_values = lyr_c1_1_out[:,f]
			#	str_colorize(s, ltr_values, aseq_gb)

			pfpl("  Y gndtrth : ")
			[ pfpl(n) for n in y[0][0].astype(numpy.uint8)]

			pfpl("  Y pred    : ")
			#pf(pred[0])
			#[ pfpl(n) for n in pred[0][0].astype(numpy.uint8)]
			#pf("")

			for loc in range(len(pred[0])-1, -1, -1):
				#pf("pred loc:", loc)
				#pf("pred[0] len:", len(pred[0][0]))
				#pf("pred[0]:", pred[0][0])
				val = int(pred[0][int(loc)]+.2)  # Add .2 for "close to 1 (true)"
				if val: # true == insert space
					s = s[:loc+1] + '/' + s[loc+2:] # use if replacing
					#s = s[:loc+1] + '/' + s[loc+1:] # use if inserting
			#pfp("  Y sentence (breaks inserted):\n", s)

			#for off in ((pred[0]*window).astype(numpy.int8)[::-1]):
			#for off in ((pred[0]*window).astype(numpy.int8)):
				#for off in (pred[0]):
				#pf(" . at", off)

	if not do_train:
		pf(yel, "Skipping training", rst)
		pf(bred, "Returning early and never running fit_generator", rst)
	if do_train:
		global fit_start_time, last_epoch_time
		#total_sets = total_wpunc = total_wopunc = 0  # Reset stats
		fit_start_time = time.time()
		last_epoch_time = time.time()

		generator = generate_texts_rnd('train')
		generator_val = generate_texts_rnd('val')

		pf("Total snippets:", total_sets)
		pf("Total snippets w/ punc:", total_wpunc)
		pf("Total snippets w/o punc:", total_wopunc)
		pf("Calling fit_generator")
		pf("Model:", model)
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
	snipverbose = 2
	snipverbose = 1
	snipverbose = 0
	if snipverbose > 1: pfp("String: {{", whi, s, rst, "}}")
	origs = s
	p = re.compile('^\w+\W'); s = p.sub("", s)  # Strip initial word (part)
	p = re.compile('^\W+\s*'); s = p.sub("", s)  # Strip non-word chars
	s = s.lower()
	# White-out(tm) punctuation and symbols
	p = re.compile('[^.!?a-z0-9 ]'); s = p.sub(" ", s)  # .!? should match below
	p = re.compile('\s+'); s = p.sub(" ", s)    # Replace all spaces with single
	s = abbrs_re.sub('\\1', s)                         #   clear . after abbreviations
	#pfp(" After: {{", yel, s, rst, "}}")
	if snipverbose > 0: pfp("  With punct:", bcya, "\n", s[0:int(window*1.2)], rst)
	glob_last_wpunct = s[:int(window*1.2)]
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
			if snipverbose > 1: pf("  Punc found at:", st-i)
			start_idx.append(st-i)
			y[st-1] = 1.0
			#i += 1
			break # only using 1 right now
	global total_sets, total_wpunc, total_wopunc
	total_sets += 1
	if len(starts):
		total_wpunc += 1
	else:
		total_wopunc += 1
	# White-out that earlier match (the EOL punctuation)
	s = p.sub(" ", s)
	if snipverbose:
		pfp(" X (w/o punct):", yel, "\n", s[0:int(window*1.1)], rst) # crop for display
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
	#if snipverbose: pf("Indexes       :", start_idx)
	start_idx = start_idx / window
	#pf("Indexes scaled:", start_idx)
	start_idx = start_idx.reshape((1,punk_max))
	#pf("Orig S:", bgre, origs[:150], rst)
	#pf("Punk locs::", bcya, start_idx, rst)
	#pf("shhh", start_idx.shape)
	#if snipverbose: pf("")
	#return s, start_idx
	y = y.reshape((1,window))
	#if snipverbose: pf("String        :", s)
	if snipverbose: pf("Y            :", y)
	#if snipverbose: pf("Indexes shape :", y.shape)
	y = [ y,
		#numpy.zeros((1,80,1)),
		#numpy.zeros((1,80,50)),
		#numpy.zeros((1,80,50)),
		#numpy.zeros((1,80,10))
	]
	return s, y, len(starts)
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
		string, y, punccnt = prep_snippet_in(data)
		#pf("Punct locs  :", punclocs)
		#pf("np len int8:", string.shape)
		#pf("np len float32:", string.shape)
		#pf("Shape x:", string.shape)
		#pf(string)
		return string, y, punccnt
	except:
		raise

class SnippetFlow(object):
    def __init__(self, offset):
        self.offset = offset

    def __call__(self):
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
			string, y, punccnt = prep_snippet_in(data)
			return string, y, punccnt
		except:
			raise

def generate_texts_flow(setname): # setname='train','val','test'
	global lastfname
	global lastfname_switch
	lastfname = None
	while True:
		iset = txtsets[setname]
		if lastfname == None or randint(100) < 20:
			iidx = randint(0, len(iset)-1)
			lastfname = iset[iidx]
			lastfname_switch = True
		x, y, punccnt = get_snippet_flow(lastfname, lastfname_switch)
		lastfname_switch = False
		#if punccnt > 2:
			#pf("X:", x)
			#pf("Y:", y)
			#pf("Ys:", y.shape)
		yield x, y

def generate_texts_rnd(setname): # setname='train','val','test'
	global lastfname
	while True:
		iset = txtsets[setname]
		iidx = randint(0, len(iset)-1)
		lastfname = iset[iidx]
		x, y, punccnt = get_snippet(lastfname)
		#if punccnt > 2:
			#pf("X:", x)
			#pf("Y:", y)
			#pf("Ys:", y.shape)
		yield x, y

init()
model = model()

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
