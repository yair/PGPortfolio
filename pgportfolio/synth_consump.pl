#!/usr/bin/perl -w
use strict;
use v5.10;

# mk_const_consumptions.pl - create a copy of a consumptions file with constant values, for testing purposes.

use Data::Dumper;
use JSON;

#my $const = 100; # basis points (const consumptions)
my $const = 0.5; # ratio (scaled consumptions)

while (<>) {
    my $origs = decode_json $_ or die "Cannot decode json: $_: $!\n";
    my %out;
    if ($const < 1) {
        %out = map { $_ => $const * $origs->{$_} } keys %$origs;
    } else {
        %out = map { $_ => $const } keys %$origs;
    }
    my $json = encode_json(\%out);
    open my $fh, '>', "./const_consumptions_$const.json" or die "Cannot open ./const_consumptions_$const.json for writing: $!\n";
    print $fh $json;
    exit;
}
