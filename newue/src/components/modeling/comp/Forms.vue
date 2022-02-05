<template>
  <div class="form-inline-label">
    <template v-for="({ type, value, list, event, label, parse, name, placeholder }, key) of items">
      <Tuple
        v-if="type === 'text_array'"
        :value="getValue(valueDef[name], value)"
        :label="label"
        type="text"
        :parse="parse"
        :name="name"
        :key="blockType + key"
        :error="getError(parse)"
        :placeholder="placeholder"
        inline
        @change="change"
      />
      <Input
        v-if="type === 'number' || type === 'text'"
        :value="getValue(valueDef[name], value)"
        :label="label"
        :type="type"
        :parse="parse"
        :name="name"
        :key="blockType + key"
        :error="getError(parse)"
        :placeholder="placeholder"
        inline
        @change="change"
      />
      <DCheckbox
        v-if="type === 'checkbox'"
        inline
        :value="getValue(valueDef[name], value)"
        :label="label"
        type="checkbox"
        :parse="parse"
        :name="name"
        :event="event"
        :error="getError(parse)"
        :key="blockType + key"
        :placeholder="placeholder"
        @change="change"
      />
      <DSelect
        v-if="type === 'select'"
        :value="valueDef[name]"
        :label="label"
        :list="list"
        :parse="parse"
        :name="name"
        :key="blockType + key"
        :error="getError(parse)"
        :placeholder="placeholder"
        @change="change"
        style="width: 200px;"
      />
    </template>
  </div>
</template>

<script>
import Input from '@/components/forms/Input.vue';
import Tuple from '@/components/forms/Tuple.vue';
// import Select from '@/components/forms/Select.vue';

export default {
  name: 'Forms',
  components: {
    Input,
    // Select,
    Tuple,
  },
  props: {
    data: {
      type: Object,
      default: () => ({ type: 'main', items: [], value: {} }),
    },
    id: Number,
  },
  computed: {
    errors() {
      return this.$store.getters['modeling/getErrorsFields'] || {};
    },
    items() {
      return this.data?.items || [];
    },
    valueDef() {
      // console.log(this.data?.value);
      return this.data?.value || {};
    },
    type() {
      return this.data?.type || '';
    },
    blockType() {
      return this.data?.blockType || '';
    },
  },
  methods: {
    getError(parse) {
      if (!this.id) return;
      // const key = parse.replace('parameters', `[${this.id}][parameters]`)
      // console.log(parse)
      // console.log(this.errors)
      // console.log(this.id)
      return this.errors?.[parse]?.[0] || '';
    },
    change(e) {
      this.$emit('change', { type: this.type, ...e });
    },
    getValue(val, defVal) {
      const value = val ?? defVal;
      // if (typeof value === "object") {
      //   return value.join();
      // }
      return value;
    },
  },
  filters: {
    toString: function (value) {
      // console.log( value )
      if (typeof value === 'object') {
        return value.join();
      }
      return value;
    },
    isCheck: function (value) {
      // console.log( value )
      // console.log( typeof(value) )
      return !!value;
    },
  },
};
</script>