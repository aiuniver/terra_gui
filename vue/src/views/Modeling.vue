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
.cont {
  display: flex;
  background: #17212b;
  padding: 0;
}

.one {
  width: 100%;
  flex-shrink: 1;
  border-right: #0e1621 1px solid;
}
.two {
  flex-shrink: 0;
  width: 400px;
}
</style>
