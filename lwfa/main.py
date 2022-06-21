
from cement import App, TestApp, init_defaults
from cement.core.exc import CaughtSignal
from .core.exc import LWFAError
from .controllers.base import Base
from .controllers.lwfa import LWFA

from nf4ip.main import NF4IP
from nf4ip.controllers.tools import Tools
import os

basePath = os.path.dirname(__file__)
projectConfig = './config'


class LWFA(NF4IP):
    """LWFA primary application."""

    class Meta:
        label = 'lwfa'

        # call sys.exit() on close
        exit_on_close = True

        # configuration handler
        config_handler = 'yaml'

        # configuration file suffix
        config_file_suffix = '.yml'

        config_dirs = [projectConfig]

        # set the log handler
        log_handler = 'logging'

        # set the output handler
        output_handler = 'jinja2'

        # register handlers
        handlers = [
            Base, Tools, LWFA
        ]

        extensions = [
            'yaml',
            'logging',
            'jinja2',
            'nf4ip.ext.inn',
            'nf4ip.ext.tensorboard',
            'nf4ip.ext.vae',
            'nf4ip.ext.progressbar'
        ]



class LWFATest(TestApp,LWFA):
    """A sub-class of LWFA that is better suited for testing."""

    class Meta:
        label = 'unit_tests'


def main():
    with LWFA() as app:
        try:
            app.run()
        except AssertionError as e:
            print('AssertionError > %s' % e.args[0])
            app.exit_code = 1

            if app.debug is True:
                import traceback
                traceback.print_exc()

        except LWFAError as e:
            print('LWFAError > %s' % e.args[0])
            app.exit_code = 1

            if app.debug is True:
                import traceback
                traceback.print_exc()

        except CaughtSignal as e:
            # Default Cement signals are SIGINT and SIGTERM, exit 0 (non-error)
            print('\n%s' % e)
            app.exit_code = 0


if __name__ == '__main__':
    main()
