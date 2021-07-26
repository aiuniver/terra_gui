<template>
  <div class="form-inline-label">
    <template v-for="({ type, value, list, event, label, parse, name }, key) of items">
      <Input
        v-if="type === 'tuple'"
        :value="value"
        :label="label"
        type="text"
        :parse="parse"
        :name="name"
        :key="key"
      />
      <Input
        v-if="type === 'number' || type === 'text'"
        :value="value"
        :label="label"
        :type="type"
        :parse="parse"
        :name="name"
        :key="key"
      />
      <Checkbox
        v-if="type === 'checkbox'"
        :value="value"
        :label="label"
        type="checkbox"
        :parse="parse"
        :name="name"
        :event="event"
        :key="key"
      />
      <Select
        v-if="type === 'select'"
        :label="label"
        :lists="list"
        :value="value"
        :parse="parse"
        :name="name"
        :key="key"
      />
    </template>
  </div>
</template>

<script>
import Input from "@/components/forms/Input.vue";
import Checkbox from "@/components/forms/Checkbox.vue";
import Select from "@/components/forms/Select.vue";

export default {
  name: "Forms",
  components: {
    Input,
    Select,
    Checkbox
  },
  props: {
    items: {
      type: Array,
      required: true,
    },
    parse: {
        type: String
    },
  },
};
</script>