const resource = 'slots';

export default $axios => ({
  all(queryObject = {}) {
    const query = new URLSearchParams(queryObject).toString();

    return $axios.$get(`${resource}${query ? '?' + query : ''}`);
  },

  show(id) {
    return $axios.$get(`${resource}/${id}`);
  },

  create(payload) {
    return $axios.$post(`${resource}`, payload);
  },

  update(id, payload) {
    return $axios.$post(`${resource}/${id}`, payload);
  },

  select(payload) {
    return $axios.$post(`select/${resource}`, payload);
  },

  delete(id) {
    return $axios.$delete(`${resource}/${id}`);
  },

  workload(date = '') {
    return $axios.$get(`workload/slots${date ? '?date=' + date : ''}`);
  }
});
