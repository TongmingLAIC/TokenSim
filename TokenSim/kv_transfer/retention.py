"""Content-bound retention declarations independent of transfer references."""

from dataclasses import dataclass
import math

from TokenSim.config.validation import integer, number
from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class RetentionClaim:
    owner: str
    revision: int
    content: object
    priority: int
    expires_at: float


class RetentionRegistry:
    def __init__(self, clock):
        self.clock = clock
        self.claims = {}
        self.owners = {}
        self.revisions = {}
        self.refreshers = {}

    def bind(self, namespace, refresh):
        self.refreshers[namespace] = refresh

    def declare(self, namespace, blocks, *, owner, revision, priority, ttl):
        integer(revision, "retention revision", minimum=1)
        integer(priority, "cache priority", minimum=0)
        number(ttl, "retention TTL", positive=True)
        if priority > 100 or not isinstance(owner, str) or not owner:
            raise ConfigurationError(
                "Retention requires an owner and priority in [0,100]"
            )
        identity = namespace, owner
        if revision <= self.revisions.get(identity, 0):
            return False
        self.revisions[identity] = revision
        now = self.clock()
        for block in blocks:
            if block.block_hash is None:
                continue
            key = namespace, block.block_number
            self.claims.setdefault(key, {})[owner] = RetentionClaim(
                owner, revision, block.block_hash, priority, now + ttl
            )
            self.owners.setdefault(owner, set()).add(key)
            self.refreshers[namespace](block.block_number)
        return True

    def priority(self, namespace, block):
        key = namespace, block.block_number
        claims = self.claims.get(key, {})
        now = self.clock()
        live = [
            claim
            for claim in claims.values()
            if claim.expires_at > now and claim.content == block.block_hash
        ]
        if len(live) != len(claims):
            if live:
                self.claims[key] = {claim.owner: claim for claim in live}
            else:
                self.claims.pop(key, None)
        return (
            max((claim.priority for claim in live), default=0),
            min((claim.expires_at for claim in live), default=math.inf),
        )

    def release(self, owner):
        for key in self.owners.pop(owner, ()):
            claims = self.claims.get(key)
            if claims is not None:
                claims.pop(owner, None)
                if not claims:
                    self.claims.pop(key, None)
                self.refreshers[key[0]](key[1])

    def reset(self):
        owners = tuple(self.owners)
        for owner in owners:
            self.release(owner)
        self.revisions.clear()
