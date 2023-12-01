#!/usr/bin/env python

import os

import click

from pyp.streampyp.web import Web


@click.group()
def cli():
    pass


@cli.command('slurm_started')
def slurm_started():
    Web().slurm_started(_arrayid())


@cli.command('slurm_ended')
@click.option('--exit', default=0)
def slurm_ended(exit):
    Web().slurm_ended(_arrayid(), exit)


@cli.command()
def ping():
    """
    Sends a simple 'ping' request to the web server, expecting a simple 'pong' response.
    Useful for testing/debugging the connection between PYP and the web server.
    """
    web = Web()
    print('sending ping to %s ...' % web.host)
    response = web.ping()
    print('response:\n%s' % response)


def _arrayid():
    # get the array id, if any
    try:
        return int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        return None


if __name__ == "__main__":
    cli()

