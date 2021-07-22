<template>
  <div class="form-inline-label">
    <template v-for="({ type, default: def, available, event }, key) of items">
      <Input
        v-if="type === 'tuple'"
        :value="def.join()"
        :label="key"
        type="text"
        :parse="`[${parse}][${key}]`"
        :name="key"
        :key="key"
      />
      <Input
        v-if="type === 'int' || type === 'string'"
        :value="def"
        :label="key"
        :type="type === 'int' ? 'number' : 'text'"
        :parse="`[${parse}][${key}]`"
        :name="key"
        :key="key"
      />
      <Checkbox
        v-if="type === 'bool'"
        :value="def"
        :label="key"
        type="checkbox"
        :parse="`[${parse}][${key}]`"
        :name="key"
        :event="event"
        :key="key"
      />
      <Select
        v-if="available"
        :label="key"
        :lists="available"
        :value="def || ''"
        :parse="`[${parse}][${key}]`"
        :name="key"
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
      type: Object,
      required: true,
    },
    parse: {
        type: String
    },
  },
};
</script>