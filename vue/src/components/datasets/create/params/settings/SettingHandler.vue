<template>
  <div>
<<<<<<< HEAD
    <template v-for="(data, index) of formsHandler">
      <TAutoFieldHandler v-bind="data" :parameters="{}" :key="index" :idKey="'key_' + index" root @change="change" />
    </template>
=======
    <div class="panel-input__forms">
      <t-field label="Название">
        <d-input-text :value="name" @change="onChangeBlock"></d-input-text>
      </t-field>
      <template v-for="(data, index) of forms">
        <t-auto-field-handler v-bind="data" :parameters="parameters" :id="id" :key="id + index" :idKey="'key_' + index" root @change="onChange" />
      </template>
    </div>
    {{ selected }}
>>>>>>> dev_dataset
  </div>
</template>

<script>
<<<<<<< HEAD
import { mapGetters } from 'vuex';
import TAutoFieldHandler from '@/components/new/blocks/TAutoFieldHandler';

export default {
  components: {
    TAutoFieldHandler
  },
=======
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
>>>>>>> dev_dataset
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