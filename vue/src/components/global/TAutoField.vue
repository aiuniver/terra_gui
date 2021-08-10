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
      <t-auto-field v-bind="data" :idKey="idKey + name + i" :key="(idKey + name) + i" @change="$emit('change', $event)"/>
    </template>
  </div>
</template>

<script>
export default {
  name: 'TAutoField',
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
      this.valueIn = null
      this.$emit('change', { value, name });
      this.$nextTick(() => {
        this.valueIn = value
      })
    },
  },
  created() {
    console.log(this.name, this.value)
    this.valueInt = this.value;
  },
  mounted() {
    this.$nextTick(() => {
        this.valueIn = this.value
    })
    this.$emit('height', this.$el.clientHeight);
  },
};
</script>