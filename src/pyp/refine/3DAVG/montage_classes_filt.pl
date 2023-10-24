#!/usr/bin/perl
use strict;
my $filecount=0;
my $tilesize=100;


# for any size
    my $llim=sprintf("%d",$tilesize*24/100);
    my $ulim=sprintf("%d",$tilesize*75/100);

#my $mydir=shift;

my $classfile=shift;
if ($classfile eq 'help' or $classfile eq ""){

    print "Usage:\n\t\tmontage_classes.pl <input classfile (no extension)> <filter=no|yes|filtername.xml> [<extra classes (mrc)>]\n";
    exit;
}


# Parse data on filter file, if a valid filename is entered, use that file
# else assume no for no input and use preformatted name for filter without
# file name provided
my $filteron=0;
my $xmlfile=shift||"no";
if ($xmlfile eq "yes") {
    sprintf("%s.flt",$classfile);
    die "$xmlfile file not found" unless (-e $xmlfile);
    $filteron=1;
}elsif ($xmlfile eq "no"){
    $filteron=0;
}else{
    die "$xmlfile file not found" unless (-e $xmlfile);
    $filteron=1;
}


my $outfile=sprintf("%s.png",$classfile);
#my $globalavg=shift;


die "$xmlfile file not found" unless (-e $xmlfile);
my $classfile=sprintf("%s_averages.txt",$classfile);

die "Attempt to overwrite $classfile" if ($classfile eq $outfile);
die "Attempt to overwrite existing output file $outfile" if (-e $outfile);

my $classcounter=0;
printf "Reading class file:\t%s\n", $classfile;
printf "Writing to file:\t%s\n", $outfile;
open (SCRIPT,">scripted.sh") or die;
open (CLASS,"<$classfile") or die "Unable to open classfile: $classfile";
unlink $outfile;
while(my $line=<CLASS>){
    chomp $line;
    my @line=split (/\t/,$line);
    next if $line[0]=~/number/;
    my $infile=$line[32];
    my $color=("lightgray","green","blue","yellow","cyan","magenta")[abs($line[31])];

    if ( $line[31] == -1 ){
    	$color="red";
    }elsif (abs($line[31])>5){
    $color="white";
    }

## Montage class
	my $command;
	if ( $filteron == 1 ){
		# Test_Metric_Filter
		$command=sprintf("Test_Metric_Filter %s -1 %s",$xmlfile,$infile);
    	warn if system $command;
    	print SCRIPT $command."\n";
    	$command=sprintf("${PYP_DIR}/IMOD_4.10/bin/newstack %s.filtered.mrc tmp1.mrc -multadd -1,0",$infile);
	} else {
    	$command=sprintf("${PYP_DIR}/IMOD_4.10/bin/newstack %s tmp1.mrc -multadd -1,0",$infile);
    }
    warn if system $command;
    print SCRIPT $command."\n";
#    my $command=sprintf("newstack tmp1.mrc tmp.mrc -float 1 -mode 0");
    my $command=sprintf("${PYP_DIR}/IMOD_4.10/bin/newstack tmp1.mrc tmp.mrc -scale 0,255 -mode 0");
    warn if system $command;
    print SCRIPT $command."\n";
    my $command=sprintf("${PYP_DIR}/IMOD_4.10/bin/mrc2tif tmp.mrc tmp");
    warn if system $command;
    print SCRIPT $command."\n";
	if ( $filteron == 1 ){
    	$command=sprintf("${PYP_DIR}/IMOD_4.10/bin/xyzproj %s.filtered.mrc tmp.mrc -axis Z -angles 0,50,10",$infile);
    } else {
    	$command=sprintf("${PYP_DIR}/IMOD_4.10/bin/xyzproj %s tmp.mrc -axis Z -angles 0,50,10",$infile);
   	}
    warn if system $command;
    print SCRIPT $command."\n";
    my $command=sprintf("${PYP_DIR}/IMOD_4.10/bin/newstack tmp.mrc tmp1.mrc -multadd -1,0");
    warn if system $command;
    print SCRIPT $command."\n";
#    my $command=sprintf("newstack tmp1.mrc tmp.mrc -float 2 -mode 0");
    my $command=sprintf("${PYP_DIR}/IMOD_4.10/bin/newstack tmp1.mrc tmp.mrc -scale 0,255 -mode 0");
    warn if system $command;    
    print SCRIPT $command."\n";
    my $command=sprintf("${PYP_DIR}/IMOD_4.10/bin/mrc2tif tmp.mrc side");
    warn if system $command;
    print SCRIPT $command."\n";
    my $command=sprintf("convert -size %dx%d -gravity Center -background %s caption:'%s' -geometry %dx%d+1+1 title.tif",
			$tilesize*10,$tilesize,$color,$infile,$tilesize*3,$tilesize);
    warn if system $command;
    print SCRIPT $command."\n";
    map{unlink sprintf("tmp.%03d.tif",$_)}(0..$llim,$ulim..$tilesize);
    my $command=sprintf("montage side.???.tif tmp.???.tif -geometry %dx%d+1+1 -tile x1 -background %s class.tif",$tilesize,$tilesize,$color);
    warn if system $command;
    print SCRIPT $command."\n";
    my $command=sprintf("convert title.tif class.tif +append class.%03d.tif",$classcounter++);
    warn if system $command;
    print SCRIPT $command."\n";
## Cleanup
    map{unlink $_}(<side.*.tif>,<tmp.*.tif>,<tmp*>,"class.tif","title.tif");
    # cleanup filtered file
 	if ( $filteron == 1 ){
    	my $command=sprintf("rm %s.filtered.mrc",$infile);
    	warn if system $command;
    	print SCRIPT $command."\n";
    }
}
close CLASSFILE;
while(my $infile=shift){
    printf "Processing also:\t%s\n",$infile; 
    my $color="green";
## Montage class  
    my $command=sprintf("${PYP_DIR}/IMOD_4.10/bin/newstack %s tmp1.mrc -multadd -1,0",$infile);
    warn if system $command;
    my $command=sprintf("${PYP_DIR}/IMOD_4.10/bin/newstack tmp1.mrc tmp.mrc -float 2 -mode 0");
    warn if system $command;
    my $command=sprintf("${PYP_DIR}/IMOD_4.10/bin/mrc2tif tmp.mrc tmp");
    warn if system $command;
    my $command=sprintf("${PYP_DIR}/IMOD_4.10/bin/xyzproj %s tmp.mrc -axis Z -angles 0,50,10",$infile);
    warn if system $command;
    my $command=sprintf("${PYP_DIR}/IMOD_4.10/bin/newstack tmp.mrc tmp1.mrc -multadd -1,0");
    warn if system $command;
    my $command=sprintf("${PYP_DIR}/IMOD_4.10/bin/newstack tmp1.mrc tmp.mrc -float 2 -mode 0");
    warn if system $command;    
    my $command=sprintf("${PYP_DIR}/IMOD_4.10/bin/mrc2tif tmp.mrc side");
    warn if system $command;
    my $command=sprintf("convert -size %dx%d -gravity Center -background %s caption:'%s' -geometry %dx%d+1+1 title.tif",
                        $tilesize*10,$tilesize,$color||"yellow",$infile,$tilesize*3,$tilesize);
    warn if system $command;
    map{unlink sprintf("tmp.%03d.tif",$_)}(0..$llim,$ulim..$tilesize);
    my $command=sprintf("montage side.???.tif tmp.???.tif -geometry %dx%d+1+1 -tile x1 -background %s class.tif",$tilesize,$tilesize,$color||"yellow");
    warn if system $command;
    my $command=sprintf("convert title.tif class.tif +append class.%03d.tif",$classcounter++);
#    my $command=sprintf("convert title.tif class.tif +append class.tif");
    warn if system $command;
### Add to file 
#    my $command=sprintf("convert %s class.tif -append %s",$outfile,$outfile);    
#    warn if system $command;
## Cleanup
    map{unlink $_}(<side.*.tif>,<tmp.*.tif>,<tmp*>,"class.tif","title.tif");
}



my $command=sprintf("convert class.???.tif -append %s",$outfile);    
warn if system $command;
print SCRIPT $command."\n";
map{unlink $_}(<class.*.tif>);

