const brackets = /(\[[^\[\]]*\])/g; //eslint-disable-line

function parse_keys(string) {
  const keys = [];
  const prefix = /^([^\[\]]*)/; //eslint-disable-line
  const children = new RegExp(brackets);
  let match = prefix.exec(string);
  if (match[1]) {
    keys.push(match[1]);
  }
  while ((match = children.exec(string)) !== null) {
    keys.push(match[1]);
  }
  return keys;
}

function hash_assign(result, keys, value) {
  if (keys.length === 0) {
    result = value;
    return result;
  }
  const key = keys.shift();
  const between = key.match(/^\[(.+?)\]$/);

  if (key === "[]") {
    result = result || [];
    if (Array.isArray(result)) {
      result.push(hash_assign(null, keys, value));
    } else {
      result._values = result._values || [];
      result._values.push(hash_assign(null, keys, value));
    }
    return result;
  }
  if (!between) {
    result[key] = hash_assign(result[key], keys, value);
  } else {
    const string = between[1];
    const index = +string;
    if (isNaN(index)) {
      result = result || {};
      result[string] = hash_assign(result[string], keys, value);
    } else {
      result = result || [];
      result[index] = hash_assign(result[index], keys, value);
    }
  }
  return result;
}
function hash_serializer(result, key, value) {
  const matches = key.match(brackets);
  if (matches) {
    const keys = parse_keys(key);
    hash_assign(result, keys, value);
  } else {
    const existing = result[key];
    if (existing) {
      if (!Array.isArray(existing)) {
        result[key] = [existing];
      }
      result[key].push(value);
    } else {
      result[key] = value;
    }
  }
  return result;
}

export default hash_serializer;
