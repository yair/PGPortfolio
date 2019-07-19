#!/usr/bin/perl -w
use strict;
use v5.10;

# TODO:
# 1. Calc and mark date boundaries of BT, to see who are comparable, and be able to mark them accordingly on the chart if they're not.

use Data::Dumper;
use JSON;

my $state = 'start';
my $model = -1;
my @coins = [];
my $step = -1;
my $totsteps = -1;
my @omegas = [];
my $omegastr = '';
my @assets = [];

while (<>) {
    if ($state eq 'start' and /hey/) {
        $state = 'hey';
    }
    if ($state eq 'hey' and /Started logging to file \.\/train_package\/(\d+)\/programlog/) {
        $model = $1;
        $state = 'model';
    }
    if ($state eq 'model' and /Selected coins are: \[(.*)\]/) {
        @coins = eval("($1)");
        $state = 'coins';
    }
    if ($state eq 'coins' and /Class name is BackTest/) {
        $state = 'bt';
    }
    if ($state eq 'bt' and /seconds before trading session (\d+)\/(\d+)$/) {
        $step = $1;
        $totsteps = $2;
    }
    if ($state eq 'bt' and /ERROR:root:NNAget: last omega: array\(\[(.*)$/) {
        $omegastr = $1;
    }
    if ($state eq 'bt' and length($omegastr) > 0 and /^\s+(\d.*\,)$/) {
        $omegastr .= $1;
    }
    if ($state eq 'bt' and length($omegastr) > 0 and /^\s+(\d.*\d)\]\)$/) {
        $omegastr .= $1;
        #        say "omegastr=$omegastr";
        $omegas[$step] = [split ',', $omegastr];
        $omegastr = '';
    }
    if ($state eq 'bt' and /total assets are (\d.*\d) BTC at/) {
        $assets[$step] = $1;
    }
    if ($state eq 'bt' and /Resetting graph and closing session/) {
        my $outs = encode_json({ 'model' => $model,
                'coins' => \@coins,
                'omegas' => \@omegas,
                'assets' => \@assets,
            });
        open my $fh, '>', "btsum/bt_summary_$model.json" or die "Cannot open btsum/bt_summary_$model.json for writing: $!\n";
        print $fh $outs;
        close $fh or die "Cannot close btsum/bt_summary_$model.json after writing: $!\n";
        say "Saved btsum/bt_summary_$model.json";
        $state = 'start';
        $model = -1;
        @coins = [];
        $step = -1;
        $totsteps = -1;
        @omegas = [];
        $omegastr = '';
        @assets = [];
    }
}

#say "model=$model";
#say "coins=(" . join(', ', @coins) . ')';
#say "omega[300][5]=" . Data::Dumper->Dump([$omegas[300]->[5]]);
#say "assets[300]=" . $assets[300];
#say "state=$state";
