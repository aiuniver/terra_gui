<template>
  <div>
    <div class="panel-input__forms">
      <t-field label="Название">
        <d-input-text :value="name" @change="onChangeBlock"></d-input-text>
      </t-field>
      <template v-for="(data, index) of forms">
        <t-auto-field-handler v-bind="data" :parameters="parameters" :id="id" :key="id + index" :idKey="'key_' + index" root @change="onChange" />
      </template>
    </div>
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
    name() {
      return this.selected?.name || '';
    },
  },
  methods: {
    ...mapActions({
      setParameters: 'create/setParameters',
      editBlock: 'create/editBlock',
    }),
    onChangeBlock({ value }) {
      const block = { ...this.selected };
      block.name = value;
      this.editBlock(block);
    },
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