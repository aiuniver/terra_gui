<template>
  <div class="field-form field-inline">
    <div><label>Train + val + test</label></div>
    <div><Range /></div>
    <div>
      <input type="number" name="parameters[train_part]" />
      <input type="number" name="parameters[val_part]" />
      <input type="number" name="parameters[test_part]" />
    </div>
  </div>
</template>

<script>
import Range from "@/components/forms/Range.vue";
export default {
  props: {},
  components: {
    Range,
  },
};
</script>