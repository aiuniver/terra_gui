<template>
  <main class="page-modeling">
    <div class="cont">
      <ModalLoadModel />
      <Toolbar />
      <ModCanvas2 />
      <Params />
    </div>
  </main>
</template>

<script>
import Toolbar from "@/components/modeling/Toolbar";
import ModCanvas2 from "@/components/modeling/ModCanvas2";
import Params from "@/components/modeling/Params";
import ModalLoadModel from "@/components/modeling/ModalLoadModel";

export default {
  name: "Modeling",
  components: {
    Toolbar,
    ModCanvas2,
    Params,
    ModalLoadModel,
  },
};
</script>

<style lang="scss" scoped>
.cont{
  background: #17212B;
  padding: 0;
  display: flex;
  height: 100%;
}

</style>
