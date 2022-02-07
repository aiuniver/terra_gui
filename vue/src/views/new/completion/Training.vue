<template>
  <div class="page-training">
    <Toolbar :collabse="collabse" />
    <Graphics @collabse="collabse = $event" />
    <Params />
  </div>
</template>

<script>
// import { mapGetters } from "vuex";
import Toolbar from '@/components/training/Toolbar.vue';
import Graphics from '@/components/new/training/Graphics.vue';
import Params from '@/components/new/training/Params.vue';

export default {
  name: 'Training',
  components: {
    Toolbar,
    Graphics,
    Params,
  },
  data: () => ({
    collabse: [],
  }),
  methods: {},
  created() {
    // console.log(this.$store.dispatch('trainings/data'))
  },
};
</script>

<style lang="scss" scoped>
.page-training {
  background: #17212b;
  padding: 0;
  display: flex;
  height: 100%;
}
</style>
