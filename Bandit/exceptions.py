class BanditError(Exception):
    """Exception carrying an integer exit code for CLI translation."""

    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code
