import imghdr
import os
from mimetypes import guess_type

from SimpleITK import GetArrayFromImage, ReadImage
from skimage.io import imread


class DirectoryLoader:
    def __init__(self, file_loader):
        self.file_loader = file_loader

    def load(self, path):
        for filename in self.__recurse(path):
            if self.file_loader.can_load(filename):
                for file in self.file_loader.load(filename):
                    yield file

    def __recurse(self, path):
        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                yield os.path.join(dirname, filename)


class FileLoader:
    def __init__(self, *base_loaders):
        self.base_loaders = base_loaders

    def load(self, path):
        for loader in self.base_loaders:
            if loader.can_load(path):
                return loader.load(path)

    def can_load(self, path):
        return any([l.can_read(path) for l in self.base_loaders])


class MimeBasedLoader:
    @staticmethod
    def match_mimetype(filename, toplevel_name=None, subtype_name=None):
        mimetype = guess_type(filename)[0]

        if not mimetype:
            return False

        mimetype = mimetype.split('/')

        res = True

        for index, name in enumerate([toplevel_name, subtype_name]):
            if name:
                res &= mimetype[index] == name

        return res


class DicomLoader(MimeBasedLoader):
    def load(self, path):
        if isinstance(path, unicode):
            path = path.encode()
    
        for image in GetArrayFromImage(ReadImage(path)):
            yield image

    def can_load(self, path):
        if not os.path.isfile(path):
            return False

        type_match = self.match_mimetype(path, subtype_name='dicom')

        with open(path, 'rb') as f:
            f.seek(0x80)
            content_match = f.read(4) == b"DICM"

        return type_match or content_match


class ImageLoader(MimeBasedLoader):
    def load(self, path):
        return [imread(path)]

    def can_load(self, path):
        return self.match_mimetype(path, toplevel_name='image') \
               and bool(imghdr.what(path))
