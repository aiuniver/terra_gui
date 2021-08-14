<template>
  <div class="forms">
    <t-input
      v-if="type === 'tuple'"
      :value="valueInt"
      :label="label"
      type="text"
      :parse="parse"
      :name="name"
      :key="name + idKey"
      inline
      @change="change"
    />
    <t-input
      v-if="type === 'number' || type === 'text'"
      :value="valueInt"
      :label="label"
      :type="type"
      :parse="parse"
      :name="name"
      :key="name + idKey"
      inline
      @change="change"
    />
    <t-checkbox
      v-if="type === 'checkbox'"
      inline
      :value="valueInt"
      :label="label"
      type="checkbox"
      :parse="parse"
      :name="name"
      :key="name + idKey"
      :event="event"
      @change="change"
    />
    <t-select
      v-if="type === 'select'"
      :value="valueInt"
      :label="label"
      :lists="list"
      :parse="parse"
      :name="name"
      :key="name + idKey"
      @change="change"
    />
    <template v-for="(data, i) of dataFields">
      <t-auto-field v-bind="data" :idKey="idKey + i" :key="idKey + i" :id="id" @change="$emit('change', $event)" />
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
  },
  data: () => ({
    valueIn: null,
  }),
  computed: {
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
    this.valueInt = this.value;
  },
  mounted() {
    this.$emit('change', { id: this.id, value: this.value, name: this.name, root: this.root });
    this.$nextTick(() => {
      this.valueIn = this.value;
    });
    this.$emit('height', this.$el.clientHeight);
  },
};
</script>