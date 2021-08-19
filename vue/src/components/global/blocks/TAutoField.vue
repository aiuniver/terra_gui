<template>
  <div class="forms">
    <t-input
      v-if="type === 'tuple'"
      :value="getValue"
      :label="label"
      type="text"
      :parse="parse"
      :name="name"
      :key="name + idKey"
      :error="error"
      inline
      @change="change"
    />
    <t-input
      v-if="type === 'number' || type === 'text'"
      :value="getValue"
      :label="label"
      :type="type"
      :parse="parse"
      :name="name"
      :key="name + idKey"
      :error="error"
      inline
      @change="change"
    />
    <t-checkbox
      v-if="type === 'checkbox'"
      inline
      :value="getValue"
      :label="label"
      type="checkbox"
      :parse="parse"
      :name="name"
      :key="name + idKey"
      :event="event"
      :error="error"
      @change="change"
    />
    <t-select
      v-if="type === 'select'"
      :value="getValue"
      :label="label"
      :lists="list"
      :parse="parse"
      :name="name"
      :key="name + idKey"
      :error="error"
      @change="change"
    />
    <template v-for="(data, i) of dataFields">
      <t-auto-field v-bind="data" :idKey="idKey + i" :key="idKey + i" :id="id" :parameters="parameters" @change="$emit('change', $event)" />
    </template>
  </div>
</template>

<script>
export default {
  name: 't-auto-field',
  props: {
    idKey: String,
    type: String,
    value: [String, Boolean, Number],
    list: Array,
    event: String,
    label: String,
    parse: String,
    name: String,
    fields: Object,
    id: Number,
    root: Boolean,
    parameters: Object
  },
  data: () => ({
    valueIn: null,
  }),
  computed: {
    getValue() {
      return this.parameters?.[this.name] ?? this.value;
    },
    errors() {
      return this.$store.getters['datasets/getErrors'](this.id);
    },
    error() {
      const key = this.name
      return this.errors?.[key]?.[0] || this.errors?.parameters?.[key]?.[0] || ''
    },
    dataFields() {
      if (!!this.fields && !!this.fields[this.valueIn]) {
        return this.fields[this.valueIn];
      } else {
        return [];
      }
    },
  },
  methods: {
    change({ value, name }) {
      this.valueIn = null;
      this.$emit('change', { id: this.id, value, name, root: this.root });
      this.$nextTick(() => {
        this.valueIn = value;
      });
    },
  },
  created() {
    // console.log(this.parameters)
  },
  mounted() {
    this.$emit('change', { id: this.id, value: this.getValue, name: this.name, root: this.root });
    // console.log(this.name, this.parameters, this.getValue)
    this.$nextTick(() => {
      this.valueIn = this.getValue;
    });
    this.$emit('height', this.$el.clientHeight);
  },
};
</script>