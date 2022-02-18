<template>
  <div>
    <template v-for="(data, index) of forms">
      <t-auto-field-handler v-bind="data" :parameters="parameters" :key="id + index" :idKey="'key_' + index" root @change="onChange" />
    </template>
    {{ selected }}
  </div>
</template>

<script>
import { mapActions } from 'vuex';
import { debounce } from '@/utils/core/utils';
export default {
  components: {},
  props: {
    selected: {
      type: Object,
      default: () => ({}),
    },
    forms: {
      type: Array,
      default: () => [],
    },
  },
  data: () => ({
    debounce: null,
  }),
  computed: {
    parameters() {
      return this?.selected?.parameters || {};
    },
    id() {
      return this.selected.id;
    },
  },
  methods: {
    ...mapActions({
      setParameters: 'create/setParameters',
    }),
    onChange(data) {
      console.log(data);
      const parameters = { ...this.parameters, [data.name]: data.value };
      this.setParameters({ id: this.id, parameters });
    },
  },
  created() {
    console.log('created');
    this.debounce = debounce;
  },
};
</script>