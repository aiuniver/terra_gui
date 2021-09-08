<template>
  <div class="form-inline-label">
    <template
      v-for="({ type, value, list, event, label, parse, name }, key) of items"
    >
      <Tuple
        v-if="type === 'text_array'"
        :value="getValue(valueDef[name], value)"
        :label="label"
        type="text"
        :parse="parse"
        :name="name"
        :key="blockType + key"
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
        inline
        @change="change"
      />
      <t-checkbox
        v-if="type === 'checkbox'"
        inline
        :value="getValue(valueDef[name], value)"
        :label="label"
        type="checkbox"
        :parse="parse"
        :name="name"
        :event="event"
        :key="blockType + key"
        @change="change"
      />
      <Select
        v-if="type === 'select'"
        :value="getValue(valueDef[name], value)"
        :label="label"
        :lists="list"
        :parse="parse"
        :name="name"
        :key="blockType + key"
        @change="change"
      />
    </template>
  </div>
</template>

<script>
import Input from "@/components/forms/Input.vue";
import Tuple from "@/components/forms/Tuple.vue";
import Select from "@/components/forms/Select.vue";

export default {
  name: "Forms",
  components: {
    Input,
    Select,
    Tuple
  },
  props: {
    data: {
      type: Object,
      default: () => ({ type: "main", items: [], value: {} }),
    },
  },
  computed: {
    items() {
      return this.data?.items || [];
    },
    valueDef() {
      // console.log(this.data?.value);
      return this.data?.value || {};
    },
    type() {
      return this.data?.type || "";
    },
    blockType() {
      return this.data?.blockType || "";
    },
  },
  methods: {
    change(e) {
      this.$emit("change", { type: this.type, ...e });
    },
    getValue(val, defVal) {
      const value = val ?? defVal;
      // if (typeof value === "object") {
      //   return value.join();
      // }
      return value
    },
  },
  filters: {
    toString: function (value) {
      // console.log( value )
      if (typeof value === "object") {
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