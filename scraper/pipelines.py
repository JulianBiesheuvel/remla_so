"""
Item pipelines.
"""

# pylint: skip-file
# mypy: ignore-errors

import psycopg


class PgPipeline:
    """
    This pipeline stores the scraped items in a Postgresql database.
    """

    def open_spider(self, spider):
        """Magically connect to the database."""
        # environment variable magic
        self.conn = psycopg.connect()

    def process_item(self, item, spider):
        """Write the item to the database."""
        with self.conn.cursor() as cur:
            cur.execute(  # insert only if not present already
                "insert into questions (id, title, tags) values (%s, %s, %s) on conflict do nothing",
                (item["id"], item["title"], item["tags"]),
            )
        self.conn.commit()
        return item

    def close_spider(self, spider):
        """Disconnect on close."""
        self.conn.close()
