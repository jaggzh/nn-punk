from __future__ import print_function # For * **
import sys

bgbla="\033[40m"
bgred="\033[41m"
bggre="\033[42m"
bgbro="\033[43m"
bgblu="\033[44m"
bgmag="\033[45m"
bgcya="\033[46m"
bggra="\033[47m"
bla="\033[30m"
red="\033[31m"
gre="\033[32m"
bro="\033[33m"
blu="\033[34m"
mag="\033[35m"
cya="\033[36m"
gra="\033[37m"
bbla="\033[30;1m"
bred="\033[31;1m"
bgre="\033[32;1m"
yel="\033[33;1m"
bblu="\033[34;1m"
bmag="\033[35;1m"
bcya="\033[36;1m"
whi="\033[37;1m"
rst="\033[0;m"

aseq_rg = [16,52,58,94,100,136,142,178,184,220]
aseq_rb = [16,52,53,89,90,126,127,163,164,200]
aseq_gb = [16,22,23,29,30,36,37,43,44,50]
aseq_r  = [16,52,88,124,160,196]
aseq_g  = [16,22,28,34,40,46]
aseq_b  = [16,17,18,19,20,21]
aseq_gr = [232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255]

def a256fg(a):
	return "\033[38;5;" + str(a) + "m";
def a256bg(a):
	return "\033[48;5;" + str(a) + "m";
def aseq_norm(seq, i): # Takes sequence and a value from [0-1]
	return seq[int((len(seq)-2) * i)+1]

# outputs colorized letters based on corresponding same-length
# list of values, using sequence codes from aseq
# skips first color (being too dark) by using aseq_norm()
def str_colorize(s, values, aseq):
	minv = min( values )
	maxv = max( values )
	for i in range(len(s)):
		val = values[i]
		norm = (val-minv)/(maxv - minv)
		ansival=aseq_norm(aseq, norm)
		ansistr=a256fg(ansival)
		print(ansistr, s[i], sep='', end='')
	print(rst)

def uncolor():
	global bgbla, bgred, bggre, bgbro, bggre, bgmag, bgcya, bggra
	global bla, red, gre, bro, gre, mag, cya, gra
	global bbla, bred, bgre, yel, bgre, bmag, bcya, whi
	global rst
	
	bgbla, bgred, bggre, bgbro, bggre, bgmag, bgcya, bggra = [""]*8
	bla, red, gre, bro, gre, mag, cya, gra = [""]*8
	bbla, bred, bgre, yel, bgre, bmag, bcya, whi = [""]*8
	rst = ""

def get_linux_terminal():
	import os
	env = os.environ
	def ioctl_GWINSZ(fd):
		try:
			import fcntl, termios, struct, os
			cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ,
		'1234'))
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

