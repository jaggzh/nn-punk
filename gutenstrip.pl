#!/usr/bin/perl
use File::Slurper qw(read_text write_text);

my $encoding = 'latin-1'; # default for Slurper anyway
my $fi = shift @ARGV;
my $fo = shift @ARGV;

if (!defined $fi) {
	print STDERR
		"Usage: gutenstrip.pl input.txt [output.txt]\n",
		"   or: gutenstrip.pl input.txt > output.txt\n",
		"Edit script if you want to change encoding from $encoding.\n";
	exit(0);
}

my $s = read_text($fi, $encoding, 1) || die "Can't read $fi: $!";
$s =~ s/^.*?START OF THIS PROJECT GUTENBERG EBOOK .*?[\r\n]//s;
$s =~ s/^\s+//;
$s =~ s/[\r\n]{2,}/\n/g; # remove blank lines

if (defined $fo) {
	write_text($fo, $s, $encoding);
} else {
	print $s;
}

