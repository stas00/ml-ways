#!/usr/bin/perl

# this script:
# 1. mirrors images and adjusts markdown to use a local copy
# 2. converts reference style image markdown (e.g StackOverflow (SO) answers) to inline style

# usage:
# tools/images2local.pl *.md

use strict;
use warnings;

use LWP::UserAgent;
use File::Copy;
use Data::Dumper;
use File::Basename;

my $agent = "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:58.0) Gecko/20100101 Firefox/58.0";
my $base = "images";

my @files = @ARGV;

# XXX: make sure we are in the root dir
for my $f (@files) {
    print "- Doc: $f\n";
    replace($f)
}

sub replace {
    my $f = shift;

    my ($file, $dir, $ext) = fileparse($f, qr/\.[^.]*/);
    #print "$file\n";

    my $prefix = "$base/$file";

    my $fi = $f;
    my $fo = "$f.new";

    open my $fh, "<$fi" or die "Can't open $fi: $!\n";
    my @lines = <$fh>;
    close $fh;

    my %refs = ();
    # mirror images, save their mapping
    foreach my $l (@lines) {
        # mirror images # XXX: more image formats?
        while ($l =~ m/(https?:.*?(?:png|jpg|jpeg|gif))/ig) {
            my $src = $1;
            my $dest = mirror($src, $prefix);
            $l =~ s/\Q$src\E/$dest/;
        }

        # image reference style parsing
        while ($l =~ m/^(\s+\[([^]]+)\]:\s+(.*))$/g){
            my $pat = $1;
            my $id  = $2;
            my $url = $3;
            $refs{$id} = $url;
            $l =~ s/\Q$pat\E//; # remove
        }
    }
    #print Dumper \%refs;

    # replace reference style with inline style of images:
    # SO examples of the used reference style
    # ![enter image description here][1]
    # [![enter image description here][2]][2]
    # and then at the end of the file:
    # [1]: http://i.stack.imgur.com/cmWO0.png
    # [2]: https://i.stack.imgur.com/Mkclz.png
    foreach (@lines) {
        while (/(\!\[([^]]*)\]\[([^]])\])/g) {
            my $pat    = $1;
            my $anchor = $2; # XXX: fixup description if none?
            my $id     = $3;
            if (exists $refs{$id}) {
                s/\Q$pat\E/![$anchor]($refs{$id})/g;
            }
        }
    }

    # done
    open my $fho, ">$fo" or die "Can't open $fo: $!\n";
    print $fho @lines;
    close $fho;

    # rename back
    File::Copy::move($fi, "$fi.bak"); # backup
    File::Copy::move($fo, $fi);
}

sub mirror {
    my ($url, $prefix) = @_;

    my $dest;
    if ($url =~ m|([^/]+)$|) {
        $dest = unique_dest("$prefix-$1"); # ensure a unique path
    }
    else {
        return '';
    }
    print "Mirror [$url] => [$dest]\n";
    my $lwp = LWP::UserAgent->new(agent => $agent, cookie_jar=>{});
    #return '';

    my $res = $lwp->mirror($url, $dest);
    unless ($res->is_success) {
        if ($res->code == 304) {
            print "File [$dest] hasn't changed - skipping\n";
            return $dest;
        }
        else {
            print "Failed to get $url: ", $res->status_line;
            return '';
        }
    }

    # don't get blocked
    sleep int rand 7;

    return $dest;
}

# find a variation on the proposed destination that's not taken already
sub unique_dest {
    my $dest = shift;

    my $try = $dest;
    my $c = 0;
    while (-e $try) {
        $c++;
        $try =~ s/(-\d+)?\.([^\.]+)$/-$c.$2/;
    }

    return $try;
}
